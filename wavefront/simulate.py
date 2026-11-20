"""Synthetic single-frame interferograms (Zernike aberrations + Gaussian bump)."""
import numpy as np

N = 128                 # image size (pixels)
FX, FY = 0.12, 0.08     # spatial carrier (cycles/pixel); assumed known
SIGMAS = (0.03, 0.10, 0.30)


def grid(n=N):
    ax = np.linspace(-1, 1, n)
    x, y = np.meshgrid(ax, ax)
    return x, y, np.hypot(x, y), np.arctan2(y, x)


def make_wavefront(rng, n=N):
    """Random wavefront phase (rad) inside the unit-disc aperture."""
    x, y, r, t = grid(n)
    mask = r <= 1.0
    c = lambda a: rng.uniform(-a, a)
    phi = (c(2.5) * (2 * r**2 - 1)                                   # defocus
           + c(1.2) * r**2 * np.cos(2 * t) + c(1.2) * r**2 * np.sin(2 * t)   # astigmatism
           + c(1.2) * (3 * r**3 - 2 * r) * np.cos(t)                 # coma
           + c(1.2) * (3 * r**3 - 2 * r) * np.sin(t)
           + c(1.5) * (6 * r**4 - 6 * r**2 + 1))                     # spherical
    cx, cy = rng.uniform(-0.6, 0.6, 2)                               # localized Gaussian bump
    s = rng.uniform(0.10, 0.25)
    amp = rng.choice([-1, 1]) * rng.uniform(0.3, 1.0)
    phi = phi + amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s**2))
    return (phi * mask).astype(np.float32), mask


def render(phi, rng, sigma, n=N):
    """I = a + b cos(2pi(fx x + fy y) + phi) + noise, with visibility and illumination gradient."""
    x, y, _, _ = grid(n)
    px, py = np.meshgrid(np.arange(n), np.arange(n))
    vis = rng.uniform(0.7, 1.0)
    g = rng.uniform(-0.1, 0.1, 2)
    a = 1 + g[0] * x + g[1] * y
    b = 0.85 * vis
    I = a + b * np.cos(2 * np.pi * (FX * px + FY * py) + phi) + rng.normal(0, sigma, phi.shape)
    return I.astype(np.float32)
