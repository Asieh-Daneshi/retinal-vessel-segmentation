"""Loading CHASE_DB1 and splitting it by child (not by image).

CHASE_DB1 has 28 fundus images: left (L) and right (R) eye of 14 children.
Both eyes of a child always go to the same split, so the test set only
contains children the model has never seen.
"""
from pathlib import Path

import numpy as np
from PIL import Image

# children 1-8 train, 9-10 validation (threshold choice), 11-14 test
SPLITS = {
    "train": list(range(1, 9)),
    "val": [9, 10],
    "test": [11, 12, 13, 14],
}


def load_case(root, child, eye, scale=0.5):
    """Return image (H, W, 3) float in [0, 1], vessel masks from both observers, FOV mask."""
    root = Path(root)
    stem = f"Image_{child:02d}{eye}"
    img = Image.open(root / f"{stem}.jpg").convert("RGB")
    gt1 = Image.open(root / f"{stem}_1stHO.png").convert("L")
    gt2 = Image.open(root / f"{stem}_2ndHO.png").convert("L")
    size = (round(img.width * scale), round(img.height * scale))
    img = np.asarray(img.resize(size, Image.BILINEAR), dtype=np.float32) / 255.0
    gt1 = np.asarray(gt1.resize(size, Image.NEAREST)) > 127
    gt2 = np.asarray(gt2.resize(size, Image.NEAREST)) > 127
    fov = fov_mask(img)
    return img, gt1, gt2, fov


def fov_mask(img, thresh=0.1):
    """CHASE_DB1 ships no field-of-view masks; the dark border is easy to threshold on the red channel."""
    return img[..., 0] > thresh


def load_split(root, split, scale=0.5):
    cases = []
    for child in SPLITS[split]:
        for eye in "LR":
            img, gt1, gt2, fov = load_case(root, child, eye, scale)
            cases.append({"name": f"{child:02d}{eye}", "img": img, "gt": gt1, "gt2": gt2, "fov": fov})
    return cases


def normalise(img):
    """Per-image standardisation, so brightness differences between photos matter less."""
    return (img - img.mean(axis=(0, 1))) / (img.std(axis=(0, 1)) + 1e-6)
