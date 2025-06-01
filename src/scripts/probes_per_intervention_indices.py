import argparse
import os

import numpy as np
import torch
from torch import Tensor


def get_top_k_indices(coef: np.ndarray, top_k: int) -> Tensor:
    coefs_torch = torch.from_numpy(coef[0]).float()
    return torch.topk(coefs_torch, k=top_k).indices


def process_coefs_dir(
    dir_with_coefs: str,
    output_dir: str,
    top_k,
    suffix_to_remove: str,
    suffix_to_add: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(dir_with_coefs):
        file_path = os.path.join(dir_with_coefs, filename)
        coefs = np.load(file_path)
        indices = get_top_k_indices(coefs, top_k)
        torch.save(
            indices,
            os.path.join(output_dir, filename.replace(suffix_to_remove, suffix_to_add)),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_with_coefs")
    parser.add_argument("--output_dir")
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--suffix_to_remove", default="_coef.npy")
    parser.add_argument("--suffix_to_add", default="_indices.pt")
    args = parser.parse_args()
    process_coefs_dir(
        args.dir_with_coefs,
        args.output_dir,
        args.top_k,
        args.suffix_to_remove,
        args.suffix_to_add,
    )
