"""Classical heterodyne Fourier-transform fringe analysis (baseline)."""
import numpy as np
from skimage.restoration import unwrap_phase
from .simulate import FX, FY


def demodulate(I, fx=FX, fy=FY, cutoff=0.07):
    """Return (wrapped, unwrapped) phase.

    Multiply by the conjugate carrier, low-pass with a radial Hann window
    (cutoff in cycles/pixel), take the angle, then unwrap (scikit-image).
    """
    n = I.shape[0]
    px, py = np.meshgrid(np.arange(n), np.arange(n))
    z = (I - 1.0) * np.exp(-2j * np.pi * (fx * px + fy * py))
    Z = np.fft.fft2(z)
    f = np.fft.fftfreq(n)
    r = np.hypot(*np.meshgrid(f, f))
    w = np.where(r < cutoff, 0.5 * (1 + np.cos(np.pi * r / cutoff)), 0.0)
    wrapped = np.angle(np.fft.ifft2(Z * w))
    return wrapped.astype(np.float32), unwrap_phase(wrapped).astype(np.float32)
