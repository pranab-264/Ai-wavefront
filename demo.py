"""python demo.py --sigma 0.1 --seed 5  -> results/demo.png"""
import argparse, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wavefront.simulate import make_wavefront, render
from wavefront.classical import demodulate
from wavefront.metrics import detrended_rmse
from wavefront.dataset import make_input

ap = argparse.ArgumentParser(); ap.add_argument("--sigma", type=float, default=0.10); ap.add_argument("--seed", type=int, default=5)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
phi, mask = make_wavefront(rng); I = render(phi, rng, a.sigma)
w, u = demodulate(I)
panels = [("Interferogram", I, "gray"), ("True phase (rad)", np.where(mask, phi, np.nan), "viridis"),
          (f"Classical error, RMSE {detrended_rmse(u, phi, mask):.3f}", np.where(mask, u - phi, np.nan), "coolwarm")]
if os.path.exists("checkpoints/unet_both.pt"):
    import torch
    from wavefront.model import UNet, predict
    ck = torch.load("checkpoints/unet_both.pt", map_location="cpu")
    net = UNet(in_ch=ck["in_ch"]); net.load_state_dict(ck["state"])
    p = predict(net, make_input(I[None], w[None], "both"), "cpu")[0]
    panels.append((f"U-Net error, RMSE {detrended_rmse(p, phi, mask):.3f}", np.where(mask, p - phi, np.nan), "coolwarm"))
fig, ax = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
for k, (t, im, cm) in zip(ax, panels):
    h = k.imshow(im, cmap=cm); k.set_title(t, fontsize=10); k.axis("off"); fig.colorbar(h, ax=k, fraction=0.046)
os.makedirs("results", exist_ok=True); fig.savefig("results/demo.png", dpi=150, bbox_inches="tight")
print("saved results/demo.png")
