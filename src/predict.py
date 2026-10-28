import numpy as np
import torch


@torch.no_grad()
def predict_image(model, norm_img, multiple=8):
    """Vessel probability map for a whole (normalised) image; pads to a size the U-Net accepts."""
    h, w, _ = norm_img.shape
    ph, pw = (-h) % multiple, (-w) % multiple
    x = np.pad(norm_img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].float()
    prob = torch.sigmoid(model(x))[0, 0].numpy()
    return prob[:h, :w]
