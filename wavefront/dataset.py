"""Dataset generation. Splits are made per wavefront; test wavefronts are re-rendered at every noise level."""
import numpy as np
from .simulate import make_wavefront, render, SIGMAS
from .classical import demodulate


def build_train_like(n, seed):
    rng = np.random.default_rng(seed)
    d = {k: [] for k in ("I", "wrapped", "classical", "phase", "mask")}
    for _ in range(n):
        phi, mask = make_wavefront(rng)
        sigma = float(np.exp(rng.uniform(np.log(0.03), np.log(0.30))))
        I = render(phi, rng, sigma)
        w, u = demodulate(I)
        for k, v in zip(d, (I, w, u, phi, mask)):
            d[k].append(v)
    return {k: np.stack(v) for k, v in d.items()}


def build_test(n, seed):
    rng = np.random.default_rng(seed)
    phis, masks = zip(*[make_wavefront(rng) for _ in range(n)])
    out = {"phase": np.stack(phis), "mask": np.stack(masks), "sigmas": np.array(SIGMAS)}
    I, W, U = [], [], []
    for s in SIGMAS:
        r = np.random.default_rng([seed, int(s * 1000)])
        Is = [render(p, r, s) for p in phis]
        res = [demodulate(i) for i in Is]
        I.append(np.stack(Is)); W.append(np.stack([a for a, _ in res])); U.append(np.stack([b for _, b in res]))
    out.update(I=np.stack(I), wrapped=np.stack(W), classical=np.stack(U))
    return out


def make_input(I, wrapped, mode="both"):
    """(N,C,H,W) float32 network input. Interferogram: (I-1)/0.85; wrapped phase: /pi."""
    ch = []
    if mode in ("both", "interferogram"):
        ch.append((I - 1.0) / 0.85)
    if mode in ("both", "wrapped"):
        ch.append(wrapped / np.pi)
    return np.stack(ch, axis=1).astype(np.float32)
