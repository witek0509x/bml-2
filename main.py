import argparse
from functools import partial
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler
from types import SimpleNamespace
from torch.optim import AdamW
import torch.nn.functional as F
from torch.nn.attention import SDPBackend
from collections import OrderedDict
from datasets import load_dataset, load_from_disk
from transformers import GPT2TokenizerFast
import torch.distributed as dist
import wandb
import os

import torch.multiprocessing as mp

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    MixedPrecision,
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)


mixed_precision_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len):
        super(EmbeddingLayer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        # x: (batch_size, seq_len)
        seq_len = x.size(1)
        positions = (
            torch.arange(seq_len, dtype=torch.long, device=x.device)
            .unsqueeze(0)
            .expand_as(x)
        )
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(positions)
        embeddings = token_embeddings + position_embeddings
        return embeddings


class AttentionLayer(nn.Module):
    def __init__(
            self,
            dmodel,
            heads,
    ):
        super(AttentionLayer, self).__init__()

        self.ln = nn.LayerNorm(dmodel)

        self.heads = heads

        self.input_projection = nn.Linear(dmodel, 3 * dmodel, bias=False)

        self.output_projection = nn.Linear(dmodel, dmodel, bias=False)

    def forward(self, x, attention_mask):
        x = self.ln(x)

        projected = self.input_projection(x)

        batch, seq_len = x.shape[:-1]
        q_chunk, k_chunk, v_chunk = torch.chunk(projected, chunks=3, dim=-1)
        query = q_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)
        key = k_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)
        value = v_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)

        with torch.nn.attention.sdpa_kernel(
                [
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH,
                ]
        ):
            attention_output = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=attention_mask,
                is_causal=True,
            )

        output = self.output_projection(attention_output.transpose(1, 2).flatten(-2))

        return output


def FeedForward(
        dmodel,
):
    return nn.Sequential(
        OrderedDict(
            [
                (
                    "ff_layernorm",
                    nn.LayerNorm(dmodel)
                ),
                (
                    "pre_relu",
                    nn.Linear(
                        dmodel,
                        4 * dmodel,
                        bias=True,
                    ),
                ),
                ("relu", nn.ReLU()),
                (
                    "post_relu",
                    nn.Linear(
                        4 * dmodel,
                        dmodel,
                        bias=True,
                    ),
                ),
            ]
        )
    )


class Block(nn.Module):

    def __init__(
            self,
            dmodel,
            heads,
    ):
        super().__init__()
        self.attention_layer = AttentionLayer(dmodel, heads)
        self.feed_forward_layer = FeedForward(dmodel)

    def forward(self, x, attention_mask):
        out_attention = self.attention_layer(x, attention_mask)
        x = x + out_attention

        out_feed_forward = self.feed_forward_layer(x)
        x = x + out_feed_forward
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_layer = EmbeddingLayer(
            config.vocab_size, config.d_model, config.max_len
        )
        self.blocks = nn.ModuleList(
            [Block(config.d_model, config.num_heads) for _ in range(config.num_layers)]
        )

        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        output = self.embedding_layer(input_ids)

        for block in self.blocks:
            output = block(output, attention_mask)

        output = self.head(output)
        return output


def collate_tokenize(tokenizer, sequence_length, data):
    text_batch = [element["text"] for element in data]
    tokenized = tokenizer(
        text_batch,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=sequence_length + 1,
    )
    input_ids = tokenized["input_ids"]
    tokenized["input_ids"] = input_ids[:, :-1]
    tokenized["target_ids"] = input_ids[:, 1:]
    tokenized["attention_mask"] = tokenized["attention_mask"][:, :-1]
    return tokenized


def get_dataloader(
        batch_size,
        sequence_length,
        split="train",
        buffer_size=10000,
        seed=42,
        num_workers=2,
        world_size=1,
        rank=0,
):
    if split == "train":
        hf_dataset = load_from_disk("/net/tscratch/people/plgkciebiera/datasets/c4/train")
    else:
        hf_dataset = load_from_disk("/net/tscratch/people/plgkciebiera/datasets/c4/validation")
    hf_dataset = hf_dataset.to_iterable_dataset(num_shards=64)
    hf_dataset = hf_dataset.shuffle(buffer_size=buffer_size, seed=seed)
    hf_dataset = hf_dataset.shard(num_shards=world_size, index=rank)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    dataloader = DataLoader(
        hf_dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_tokenize, tokenizer, sequence_length),
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return dataloader


def calculate_valid_loss(model, valid_dataloader, rank, validation_steps):
    valid_losses = []
    for _, batch in zip(range(validation_steps), valid_dataloader):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(rank)
            target_ids = batch["target_ids"].to(rank)
            attention_mask = batch["attention_mask"]
            outputs = model(input_ids)
            mask_loss = F.cross_entropy(
                outputs.flatten(0, -2),
                target_ids.reshape(-1).long(),
                reduction="none",
            )
            mask_loss = mask_loss[attention_mask.reshape(-1) == 1]
            loss = mask_loss.mean().item()
            valid_losses.append(loss)
            mean_valid_loss = sum(valid_losses) / validation_steps
    return mean_valid_loss


def train_model(config, rank, world_size):
    dataloader = get_dataloader(config.batch_size, config.seq_length, world_size=world_size, rank=rank)
    valid_dataloader = get_dataloader(config.batch_size, config.seq_length, split="validation", world_size=world_size, rank=rank)
    validation_steps = int(1e06 // (config.batch_size * config.seq_length))
    model = Transformer(config)
    model = FSDP(model, device_id=rank, mixed_precision=mixed_precision_policy).to(rank)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scaler = GradScaler()
    model.train()

    for i, batch in zip(range(config.train_steps), dataloader):
        input_ids = batch["input_ids"].to(rank)
        target_ids = batch["target_ids"].to(rank)
        attention_mask = batch["attention_mask"]

        if i < 4:
            print("INPUT:", rank, "      ", input_ids)

        optimizer.zero_grad()
        with autocast():
            outputs = model(input_ids)

            mask_loss = F.cross_entropy(
                outputs.flatten(0, -2),
                target_ids.reshape(-1).long(),
                reduction="none",
            )
            mask_loss = mask_loss[attention_mask.reshape(-1) == 1]
            loss = mask_loss.mean()

        wandb.log({"train_loss": loss.item(), "step": i})

        if i % config.log_valid_loss_freq == 0:
            valid_loss = calculate_valid_loss(model, valid_dataloader, rank, validation_steps)
            wandb.log({"valid_loss": valid_loss, "step": i})

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        dist.barrier()


    if rank == 0:
        final_valid_loss = calculate_valid_loss(model, valid_dataloader, rank, validation_steps)
        wandb.log({"final_valid_loss": final_valid_loss})
    wandb.finish()


def main(rank, world_size, args):

    config = SimpleNamespace(
        train_steps=args.n_training_steps,
        vocab_size=50257,
        max_len=256,
        d_model=args.dmodel,
        num_heads=args.n_heads,
        num_layers=args.n_layers,
        learning_rate=1e-4,
        dropout=0.0,
        seq_length=256,
        batch_size=args.batch_size,
        log_train_loss_freq=100,
        log_valid_loss_freq=100
    )
    dist.init_process_group("nccl", rank=rank, world_size=2)
    torch.cuda.set_device(rank)
    train_model(config, rank, world_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FSDP implementation')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--n_layers', type=int, default=4, metavar='N',
                        help='number of transformer layers (default: 4)')
    parser.add_argument('--dmodel', type=int, default=256, metavar='N',
                        help='model dimension (default: 256)')
    parser.add_argument('--n_training_steps', type=int, default=1000, metavar='LR',
                        help='number of training steps (default: 1000)')
    parser.add_argument('--n_heads', type=int, default=4, metavar='M',
                        help='Number of attention heads (default: 4)')
    args = parser.parse_args()

    wandb.login(key=os.environ['WANDB_API_KEY'])
    wandb.init(project="transformer-training", config=args.__dict__)

    world_size = torch.cuda.device_count()
    rank = int(os.environ["LOCAL_RANK"])
    print(f"WORLD_SIZE = {world_size}")
    main(rank, world_size, args)
