from argparse import ArgumentParser

import torch

from src.probes.probes import SimpleNetwork
import matplotlib.pyplot as plt



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--weight_path", "-w", default="SAE_latents/ra3taee1/checkpoints/epoch=49-step=3450.ckpt")
    parser.add_argument("--sae_type", "-s", choices=["pair", "non_pair"], default="pair")
    parser.add_argument("--top_k", "-k", default=10, type=int)
    parser.add_argument("--multiplier", "-m", default=3, type=int)
    parser.add_argument("--device", "-d", default="cuda")
    args = parser.parse_args()
    path = args.weight_path
    k = args.top_k
    multiplier = args.multiplier
    device = args.device
    sae_type = args.sae_type
    model = SimpleNetwork.load_from_checkpoint(path)
    weights = list(model.model.parameters())[0].data.to(device)
    print(weights.shape)
    top_k = torch.topk(weights, k).indices.to(device)
    # torch.save(top_k, "non_pair_top_10.pt")
    print(top_k)
    w = weights.squeeze()# [4200]
    abs_w = torch.abs(w)
    sorted_w, _ = torch.sort(w, descending=True)  # sorted magnitudes

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    axes[0][0].boxplot(sorted_w.tolist())
    axes[0][0].set_ylabel("value of activation")
    axes[0][0].set_xlabel("")
    axes[0][0].set_title("box plot of activation values")

    axes[0][1].plot(sorted_w.tolist())
    axes[0][1].set_ylabel("sorted activation values")
    axes[0][1].set_xlabel("idx of feature")
    axes[0][1].set_title("plot of activation values")


    axes[1][0].plot(sorted_w.tolist()[-20:])
    axes[1][0].set_title("sorted activation values, lowest 20")
    axes[1][0].set_xlabel("idx of feature")
    axes[1][0].set_xticklabels(list(range(sorted_w.shape[0]-20, sorted_w.shape[0])))

    axes[1][1].plot(sorted_w.tolist()[:20])
    axes[1][1].set_title("sorted activation values, highest 20")
    axes[1][1].set_xlabel("idx of feature")
    # plt.tight_layout()

    fig.suptitle(f"{sae_type}", fontsize=20)
    fig.tight_layout()
    plt.show()
