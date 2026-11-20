import numpy as np


def detrended_rmse(pred, truth, mask):
    """Aperture-restricted RMSE (rad) after removing piston and tilt of the residual."""
    n = pred.shape[0]
    ax = np.linspace(-1, 1, n)
    x, y = np.meshgrid(ax, ax)
    d = (pred - truth)[mask]
    A = np.stack([np.ones(d.size), x[mask], y[mask]], axis=1)
    coef, *_ = np.linalg.lstsq(A, d, rcond=None)
    return float(np.sqrt(np.mean((d - A @ coef) ** 2)))
