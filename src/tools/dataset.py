import os
from typing import Tuple
import random

import torch
from datasets import IterableDataset, Dataset, concatenate_datasets
from sklearn.model_selection import train_test_split


def remove_dead_features(ds: Dataset | IterableDataset, feature_name: str, masking_value: int = 0,
                         threshold: float = 1e-3) -> Dataset | IterableDataset:
    # find indices where at least one sample have value greater than threshold
    mask = torch.tensor([False] * next(iter(ds))[feature_name].shape[0])
    for example in ds:
        mask = torch.logical_or(mask, example[feature_name] > threshold)
    mask = torch.logical_not(mask)  # set True where none sample has larger value aka dead neuron
    for example in ds:
        example[feature_name][mask] = masking_value
    return ds

def load_ds_from_dirs(path: str, columns, dtype, n_shards_per_timestep: int | None = None) -> Dataset:
    datasets = []
    for timestep_dir_name in os.listdir(path):
        timestep_dir_path = os.path.join(path, timestep_dir_name)
        ds_dir_names = os.listdir(timestep_dir_path)
        if n_shards_per_timestep:
            ds_dir_names = random.sample(ds_dir_names, n_shards_per_timestep)
        for example_dir_name in ds_dir_names:
            example_dir_path = os.path.join(timestep_dir_path, example_dir_name)
            ds = Dataset.load_from_disk(example_dir_path, keep_in_memory=False)
            ds.set_format(
                type="torch",
                columns=columns,
                dtype=dtype
            )
            datasets.append(ds)
        print(f"processed {timestep_dir_name}")
    return concatenate_datasets(datasets)


def create_iterable_dataset(paths: list[str], dtype, columns: list[str]) -> IterableDataset:
    def gen(paths, dtype):
        for path in paths:
            ds = Dataset.load_from_disk(path)
            ds.set_format(
                type="torch",
                columns=columns,
                dtype=dtype
            )
            for example in ds:
                yield example

    return IterableDataset.from_generator(generator=gen, gen_kwargs={"paths": paths, "dtype": dtype})


def get_shards_paths(base_dir):
    all_directories = os.listdir(base_dir)
    shards_paths = []
    for timestep_dir_name in sorted(all_directories):
        timestrep_dir = os.path.join(base_dir, timestep_dir_name)
        for design_der_name in os.listdir(timestrep_dir):
            design_dir = os.path.join(timestrep_dir, design_der_name)
            shards_paths.append(design_dir)
    return shards_paths


def load_datasets_from_dir_of_dirs(base_dir, dtype, columns: list[str]):
    print(f"loading iterable dataset {base_dir}")
    return create_iterable_dataset(get_shards_paths(base_dir), dtype, columns)


def create_train_test_iterable_datasets(base_dir, dtype, columns: list[str], test_size=0.2) -> Tuple[
    IterableDataset, IterableDataset]:
    shards_paths = get_shards_paths(base_dir)
    train_shards, test_shards = train_test_split(shards_paths, test_size=test_size)
    return (
        create_iterable_dataset(train_shards, dtype, columns),
        create_iterable_dataset(test_shards, dtype, columns),
    )
