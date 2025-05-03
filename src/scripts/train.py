"""
Train sparse autoencoders on activations from a diffusion model.
"""

import os
import random
import sys
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import torch
import torch.distributed as dist
from datasets import Dataset, concatenate_datasets
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


def load_datasets_from_dir_of_dirs(base_dir, dtype=torch.float32, random_n_activations_from_dataset= None):
    """
    Load and concatenate datasets from multiple directories.

    Args:
        base_dirs (list[str]): List of base directory paths containing the datasets
        hookpoint (str): Name of the hookpoint directory
        dtype: Data type for the tensors (default: torch.float32)
        random_n_activations_from_dataset (int | None): sample n random activation from dataset

    Returns:
        Dataset: Concatenated dataset
    """
    datasets = []
    print(f"Concatenating datasets from {base_dir}")

    all_directories = os.listdir(base_dir)
    idx = 1
    for timestep_dir_name in sorted(all_directories):
        timestrep_dir = os.path.join(base_dir, timestep_dir_name)
        for design_der_name in os.listdir(timestrep_dir):
            design_dir = os.path.join(timestrep_dir, design_der_name)
            dataset = Dataset.load_from_disk(
               design_dir, keep_in_memory=False
            )

            # Set format for each dataset
            dataset.set_format(
                type="torch",
                columns=["values"],
                dtype=dtype,
            )
            # dataset = dataset.select(range(min(1000, len(dataset))))
            if random_n_activations_from_dataset:
                dataset = dataset.select(random.sample(range(0, len(dataset)), random_n_activations_from_dataset))
            datasets.append(dataset)
        print(f"processed {timestrep_dir}")
    # Concatenate all datasets
    return concatenate_datasets(datasets)


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
        dataset.set_format(
            type="torch",
            columns=["values"],
            dtype=dtype,
        )
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
