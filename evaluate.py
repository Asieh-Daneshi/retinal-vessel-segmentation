"""Evaluate the trained U-Net on the held-out test children.

Usage: python evaluate.py --data data/CHASEDB1 --run runs/unet
- picks the decision threshold on the validation children (never on test)
- reports Dice, sensitivity, specificity and AUC per test image
- compares with the second human observer, as a ceiling for "good enough"
- checks calibration and saves figures of the best and worst test images
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import load_split, normalise
from src.metrics import calibration, dice, scores
from src.model import UNet
from src.predict import predict_image


def error_map(pred, gt, fov):
    """White: correct vessel, red: false positive, blue: missed vessel."""
    rgb = np.zeros(gt.shape + (3,))
    rgb[pred & gt] = 1
    rgb[pred & ~gt] = (1, 0.2, 0.2)
    rgb[~pred & gt] = (0.2, 0.5, 1)
    rgb[~fov] = 0.15
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/CHASEDB1")
    ap.add_argument("--run", default="runs/unet")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--figs", default="figures")
    args = ap.parse_args()
    figs = Path(args.figs)
    figs.mkdir(exist_ok=True)

    model = UNet()
    model.load_state_dict(torch.load(Path(args.run) / "best.pt", map_location="cpu"))
    model.eval()

    val = load_split(args.data, "val", args.scale)
    test = load_split(args.data, "test", args.scale)
    for c in val + test:
        c["prob"] = predict_image(model, normalise(c["img"]).astype(np.float32))

    # 1) threshold chosen on validation children only
    grid = np.round(np.arange(0.2, 0.81, 0.05), 2)
    val_dice = [np.mean([dice(c["prob"] > t, c["gt"], c["fov"]) for c in val]) for t in grid]
    thr = float(grid[int(np.argmax(val_dice))])

    # 2) test metrics, and the second observer against the first
    rows = []
    for c in test:
        s = scores(c["prob"], c["gt"], c["fov"], thr)
        s["observer2_dice"] = dice(c["gt2"], c["gt"], c["fov"])
        s["image"] = c["name"]
        rows.append(s)
    keys = ["dice", "sensitivity", "specificity", "auc", "observer2_dice"]
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    std = {k: float(np.std([r[k] for r in rows])) for k in keys}

    # 3) calibration on test pixels inside the field of view
    conf, acc, ece = calibration([c["prob"][c["fov"]] for c in test], [c["gt"][c["fov"]] for c in test])

    print(f"threshold (from validation): {thr}")
    print(f"{'image':>6} {'Dice':>6} {'Sens':>6} {'Spec':>6} {'AUC':>6} {'Obs2':>6}")
    for r in rows:
        print(f"{r['image']:>6} {r['dice']:6.3f} {r['sensitivity']:6.3f} {r['specificity']:6.3f} {r['auc']:6.3f} {r['observer2_dice']:6.3f}")
    print("  mean " + " ".join(f"{mean[k]:6.3f}" for k in keys))
    print(f"expected calibration error: {ece:.3f}")

    json.dump({"threshold": thr, "per_image": [{k: (float(v) if k != 'image' else v) for k, v in r.items()} for r in rows],
               "mean": mean, "std": std, "ece": ece}, open(Path(args.run) / "test_results.json", "w"), indent=1)

    # figures: best and worst test image
    order = np.argsort([r["dice"] for r in rows])
    for tag, i in [("best", order[-1]), ("worst", order[0])]:
        c, r = test[i], rows[i]
        pred = c["prob"] > thr
        fig, ax = plt.subplots(1, 4, figsize=(16, 4.9))
        ax[0].imshow(c["img"]); ax[0].set_title(f"image {c['name']}")
        ax[1].imshow(c["gt"], cmap="gray"); ax[1].set_title("observer 1 (ground truth)")
        ax[2].imshow(c["prob"], cmap="gray", vmin=0, vmax=1); ax[2].set_title("U-Net probability")
        ax[3].imshow(error_map(pred, c["gt"], c["fov"])); ax[3].set_title(f"errors (Dice {r['dice']:.3f})\nred: false positive, blue: missed")
        for a in ax:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(figs / f"test_{tag}.png", dpi=110)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.plot(conf, acc, "o-", label=f"U-Net (ECE {ece:.3f})")
    ax.set_xlabel("predicted vessel probability")
    ax.set_ylabel("observed vessel fraction")
    ax.set_title("Calibration on test children")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "calibration.png", dpi=120)


if __name__ == "__main__":
    main()
