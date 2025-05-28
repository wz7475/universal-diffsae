import argparse
import os

import torch

from src.tools.dataset import load_ds_from_dirs, get_dataset_latents_target_label

def build_path(base_dir: str, sae_type: str) -> str:
    return os.path.join(base_dir, "latents", sae_type)

def main(input_path, output_dir, feature_type) -> None:
    raw_ds = load_ds_from_dirs(input_path, [feature_type, "values"], torch.float32)
    for label in set(raw_ds[feature_type]):
        print(f"processing {label}")
        labels_ds = get_dataset_latents_target_label(raw_ds, feature_type, label)
        labels_ds.reset_format()
        labels_ds.save_to_disk(os.path.join(output_dir, label))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_shards_path", type=str, required=True)
    parser.add_argument("--output_datasets_dir", type=str, required=True)
    parser.add_argument("--sae_type", choices=["pair", "non_pair"], default="pair")
    parser.add_argument("--feature_type", type=str, default="subcellular")
    args = parser.parse_args()

    main(build_path(args.dataset_shards_path, args.sae_type), args.output_datasets_dir, args.feature_type)
