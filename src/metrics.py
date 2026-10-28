"""Segmentation metrics, all computed inside the field of view only."""
import numpy as np
from sklearn.metrics import roc_auc_score


def confusion(pred, gt, fov):
    p, g = pred[fov], gt[fov]
    return np.sum(p & g), np.sum(p & ~g), np.sum(~p & g), np.sum(~p & ~g)


def dice(pred, gt, fov):
    tp, fp, fn, _ = confusion(pred, gt, fov)
    return 2 * tp / (2 * tp + fp + fn)


def scores(prob, gt, fov, thr):
    pred = prob > thr
    tp, fp, fn, tn = confusion(pred, gt, fov)
    return {
        "dice": 2 * tp / (2 * tp + fp + fn),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "auc": roc_auc_score(gt[fov], prob[fov]),
    }


def calibration(probs, gts, bins=10):
    """Reliability data: mean predicted probability vs observed vessel fraction per bin, and ECE."""
    p = np.concatenate(probs)
    g = np.concatenate(gts).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    conf, acc, frac = [], [], []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        conf.append(p[m].mean()); acc.append(g[m].mean()); frac.append(m.mean())
    conf, acc, frac = map(np.array, (conf, acc, frac))
    return conf, acc, float(np.sum(frac * np.abs(conf - acc)))
