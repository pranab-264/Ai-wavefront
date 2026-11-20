"""Final evaluation on the untouched test set: classical vs trained U-Nets, paired stats."""
import os
import numpy as np
from scipy import stats
from wavefront.metrics import detrended_rmse
from wavefront.dataset import make_input

te = np.load("data/test.npz")
sig, phase, mask = te["sigmas"], te["phase"], te["mask"]
n = len(phase)
res = {"classical": np.array([[detrended_rmse(te["classical"][s, i], phase[i], mask[i]) for i in range(n)]
                              for s in range(len(sig))])}

try:
    import torch
    from wavefront.model import UNet, predict
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for mode in ("both", "interferogram", "wrapped"):
        path = f"checkpoints/unet_{mode}.pt"
        if not os.path.exists(path):
            continue
        ck = torch.load(path, map_location=dev)
        model = UNet(in_ch=ck["in_ch"]).to(dev); model.load_state_dict(ck["state"])
        rows = []
        for s in range(len(sig)):
            p = predict(model, make_input(te["I"][s], te["wrapped"][s], mode), dev)
            rows.append([detrended_rmse(p[i], phase[i], mask[i]) for i in range(n)])
        res[f"unet_{mode}"] = np.array(rows)
except ImportError:
    print("PyTorch not installed: classical baseline only.")

os.makedirs("results", exist_ok=True)
lines = ["method,sigma,n,mean,std,ci95_low,ci95_high,wilcoxon_p_vs_classical"]
for name, r in res.items():
    for s, sg in enumerate(sig):
        m, sd = r[s].mean(), r[s].std(ddof=1)
        lo, hi = stats.t.interval(0.95, n - 1, loc=m, scale=sd / np.sqrt(n))
        p = "" if name == "classical" else f"{stats.wilcoxon(r[s], res['classical'][s]).pvalue:.3g}"
        lines.append(f"{name},{sg},{n},{m:.4f},{sd:.4f},{lo:.4f},{hi:.4f},{p}")
open("results/results.csv", "w").write("\n".join(lines))
print("\n".join(lines))
