import os
from typing import Tuple

from datasets import IterableDataset, Dataset
from sklearn.model_selection import train_test_split


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


def create_train_test_iterable_datasets(base_dir, dtype, columns: list[str], test_size=0.2) -> Tuple[IterableDataset, IterableDataset]:
    shards_paths = get_shards_paths(base_dir)
    train_shards, test_shards = train_test_split(shards_paths, test_size=test_size)
    return (
        create_iterable_dataset(train_shards, dtype, columns),
        create_iterable_dataset(test_shards, dtype, columns),
    )

