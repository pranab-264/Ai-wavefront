"""python train.py --inputs both|interferogram|wrapped"""
import argparse, os
import numpy as np, torch, torch.nn.functional as F
from wavefront.dataset import make_input
from wavefront.model import UNet, predict
from wavefront.metrics import detrended_rmse

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", default="both", choices=["both", "interferogram", "wrapped"])
ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
tr, va = np.load("data/train.npz"), np.load("data/val.npz")
X = torch.from_numpy(make_input(tr["I"], tr["wrapped"], a.inputs))
Y = torch.from_numpy(tr["phase"])[:, None]
M = torch.from_numpy(tr["mask"].astype(np.float32))[:, None]
Xv = make_input(va["I"], va["wrapped"], a.inputs)

model = UNet(in_ch=X.shape[1]).to(dev)
print(f"params: {sum(p.numel() for p in model.parameters()):,} | device: {dev}")
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
best = 1e9
os.makedirs("checkpoints", exist_ok=True)
for ep in range(a.epochs):
    model.train()
    perm = torch.randperm(len(X))
    for i in range(0, len(X), a.bs):
        idx = perm[i:i + a.bs]
        x, y, m = X[idx].to(dev), Y[idx].to(dev), M[idx].to(dev)
        loss = (F.smooth_l1_loss(model(x), y, reduction="none") * m).sum() / m.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()
    pv = predict(model, Xv, dev)
    val = np.mean([detrended_rmse(pv[i], va["phase"][i], va["mask"][i]) for i in range(len(pv))])
    if val < best:  # checkpoint selection on validation RMSE only
        best = val
        torch.save({"state": model.state_dict(), "in_ch": X.shape[1], "inputs": a.inputs}, f"checkpoints/unet_{a.inputs}.pt")
    print(f"epoch {ep + 1:02d}  loss {loss.item():.4f}  val RMSE {val:.4f}  best {best:.4f}")
