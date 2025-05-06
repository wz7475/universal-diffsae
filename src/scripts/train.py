"""
Train sparse autoencoders on activations from a diffusion model.
"""

import os
import sys
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import torch
import torch.distributed as dist
from datasets import Dataset, IterableDataset
from simple_parsing import parse

from src.sae.config import TrainConfig
from src.sae.trainer import SaeTrainer


@dataclass
class RunConfig(TrainConfig):
    mixed_precision: str = "no"

    max_examples: int | None = None
    """Maximum number of examples to use for training."""

    seed: int = 42
    """Random seed for shuffling the dataset."""
    device: str = "cuda"
    num_epochs: int = 1
    n_random_activation: Optional[int] = None
    max_trainer_steps: Optional[int] = None

def create_iterable_dataset(paths: list[str], dtype) -> IterableDataset:
    def gen(paths, dtype):
        for path in paths:
            ds = Dataset.load_from_disk(path)
            ds.set_format(
            type="torch",
            columns=["values"],
            dtype=dtype
        )
            for example in ds:
                yield example

    return IterableDataset.from_generator(generator=gen, gen_kwargs={"paths": paths, "dtype": dtype})



def load_datasets_from_dir_of_dirs(base_dir, dtype=torch.float32, random_n_activations_from_dataset= None):

    print(f"loading iterable dataset {base_dir}")

    all_directories = os.listdir(base_dir)
    shards_paths = []
    for timestep_dir_name in sorted(all_directories):
        timestrep_dir = os.path.join(base_dir, timestep_dir_name)
        for design_der_name in os.listdir(timestrep_dir):
            design_dir = os.path.join(timestrep_dir, design_der_name)
            shards_paths.append(design_dir)
    return create_iterable_dataset(shards_paths, dtype)


def run():
    local_rank = os.environ.get("LOCAL_RANK")
    ddp = local_rank is not None
    rank = int(local_rank) if ddp else 0

    if ddp:
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group("nccl")

        if rank == 0:
            print(f"Using DDP across {dist.get_world_size()} GPUs.")

    args = parse(RunConfig)
    # add output_or_diff to the run name
    args.run_name = args.run_name + f"lr{args.lr}_{''.join(args.dataset_path[0].split('/'))}_{args.hookpoints[0]}"

    dtype = torch.float32
    if args.mixed_precision == "fp16":
        dtype = torch.float16
    elif args.mixed_precision == "bf16" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    args.dtype = dtype
    print(f"Training in {dtype=}")
    # Awkward hack to prevent other ranks from duplicating data preprocessing
    dataset_dict = {}
    if not ddp or rank == 0:
        dataset = load_datasets_from_dir_of_dirs(os.path.join(args.dataset_path[0], args.hookpoints[0]), dtype)
        dataset = dataset.shuffle(args.seed)
        if limit := args.max_examples:
            dataset = dataset.select(range(limit))
        dataset_dict[args.hookpoints[0]] = dataset


    # Prevent ranks other than 0 from printing
    with nullcontext() if rank == 0 else redirect_stdout(None):
        trainer = SaeTrainer(args, dataset_dict)

        trainer.fit(args.max_trainer_steps)


if __name__ == "__main__":
    run()
