import argparse
import os

import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.probes.probes import SimpleNetwork
from src.tools.dataset import create_train_test_iterable_datasets


def main(
        base_dir,
        sae_type,
        feature_type,
        target_label,
        train_batch_size,
        test_batch_size,
        test_size,
        cuda_devices,
        wandb_project,
        max_epochs,
        log_every_n_steps,
        lr,
):
    path = os.path.join(base_dir, "latents", sae_type)
    train_ds, test_ds = create_train_test_iterable_datasets(path, torch.float32, columns=["values", feature_type],
                                                            test_size=test_size)
    train_ds = train_ds.map(
        lambda example: {feature_type: torch.tensor((example[feature_type] == target_label), dtype=torch.float32)})
    test_ds = test_ds.map(
        lambda example: {feature_type: torch.tensor((example[feature_type] == target_label), dtype=torch.float32)})

    train_dataloader = DataLoader(train_ds, batch_size=train_batch_size)
    test_dataloader = DataLoader(test_ds, batch_size=test_batch_size)

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    logger = WandbLogger(project=wandb_project)
    model_input_dim = next(iter(train_ds))["values"].shape[0]
    model = SimpleNetwork(model_input_dim, lr)
    trainer = L.Trainer(max_epochs=max_epochs, logger=logger, log_every_n_steps=log_every_n_steps)
    trainer.fit(model, train_dataloader, test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SimpleNetwork on SAE latents")
    parser.add_argument("--base_dir", type=str, default="/data/wzarzecki/ds_sae_latents_50x")
    parser.add_argument("--sae_type", type=str, default="pair")
    parser.add_argument("--feature_type", type=str, default="subcellular")
    parser.add_argument("--target_label", type=str, default="Cytoplasm")
    parser.add_argument("--train_batch_size", type=int, default=4096)
    parser.add_argument("--test_batch_size", type=int, default=4096)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--cuda_devices", type=str, default="1")
    parser.add_argument("--wandb_project", type=str, default="SAE_latents")
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()
    main(
        args.base_dir,
        args.sae_type,
        args.feature_type,
        args.target_label,
        args.train_batch_size,
        args.test_batch_size,
        args.test_size,
        args.cuda_devices,
        args.wandb_project,
        args.max_epochs,
        args.log_every_n_steps,
        args.lr,
    )
