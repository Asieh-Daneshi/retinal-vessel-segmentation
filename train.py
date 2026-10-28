"""Train a small U-Net for retinal vessel segmentation on CHASE_DB1.

Usage: python train.py --data data/CHASEDB1 --iters 3000
Trains on random patches from the training children and keeps the checkpoint
with the best validation Dice.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data import load_split, normalise
from src.model import UNet
from src.predict import predict_image
from src.metrics import dice


def random_patch(cases, size, rng):
    c = cases[rng.integers(len(cases))]
    h, w = c["gt"].shape
    y, x = rng.integers(0, h - size), rng.integers(0, w - size)
    img = c["norm"][y:y + size, x:x + size]
    gt = c["gt"][y:y + size, x:x + size].astype(np.float32)
    k = rng.integers(4)  # random rotation and flip
    img, gt = np.rot90(img, k), np.rot90(gt, k)
    if rng.random() < 0.5:
        img, gt = img[:, ::-1], gt[:, ::-1]
    return np.ascontiguousarray(img.transpose(2, 0, 1)), np.ascontiguousarray(gt[None])


def soft_dice_loss(logits, target):
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    return 1 - (2 * inter + 1) / (p.sum() + target.sum() + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/CHASEDB1")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--out", default="runs/unet")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train = load_split(args.data, "train", args.scale)
    val = load_split(args.data, "val", args.scale)
    for c in train + val:
        c["norm"] = normalise(c["img"]).astype(np.float32)

    model = UNet()
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)

    best, log, t0 = -1.0, [], time.time()
    for it in range(1, args.iters + 1):
        model.train()
        batch = [random_patch(train, args.patch, rng) for _ in range(args.batch)]
        x = torch.from_numpy(np.stack([b[0] for b in batch]))
        y = torch.from_numpy(np.stack([b[1] for b in batch]))
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y) + soft_dice_loss(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if it % 250 == 0 or it == args.iters:
            model.eval()
            scores = [dice(predict_image(model, c["norm"]) > 0.5, c["gt"], c["fov"]) for c in val]
            vd = float(np.mean(scores))
            log.append({"iter": it, "loss": loss.item(), "val_dice": vd, "minutes": (time.time() - t0) / 60})
            print(f"iter {it:5d}  loss {loss.item():.3f}  val Dice {vd:.3f}  ({log[-1]['minutes']:.1f} min)", flush=True)
            if vd > best:
                best = vd
                torch.save(model.state_dict(), out / "best.pt")
    json.dump({"args": vars(args), "log": log, "best_val_dice": best}, open(out / "train_log.json", "w"), indent=1)


if __name__ == "__main__":
    main()
