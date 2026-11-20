import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Compact U-Net (~1.9M params): widths 32/64/128/256, 16x16 bottleneck for 128x128 input."""

    def __init__(self, in_ch=2, widths=(32, 64, 128, 256), out_scale=3.0):
        super().__init__()
        w1, w2, w3, w4 = widths
        self.e1, self.e2, self.e3, self.bott = block(in_ch, w1), block(w1, w2), block(w2, w3), block(w3, w4)
        self.d3, self.d2, self.d1 = block(w4 + w3, w3), block(w3 + w2, w2), block(w2 + w1, w1)
        self.out = nn.Conv2d(w1, 1, 1)
        self.out_scale = out_scale  # output scaled by 3 for training stability

    def forward(self, x):
        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        b = self.bott(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([up(b), e3], 1))
        d2 = self.d2(torch.cat([up(d3), e2], 1))
        d1 = self.d1(torch.cat([up(d2), e1], 1))
        return self.out_scale * self.out(d1)


@torch.no_grad()
def predict(model, X, device, bs=32):
    model.eval()
    out = [model(torch.from_numpy(X[i:i + bs]).to(device)).cpu().numpy()[:, 0] for i in range(0, len(X), bs)]
    return np.concatenate(out)
