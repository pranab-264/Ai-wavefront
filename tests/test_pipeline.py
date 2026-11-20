import numpy as np, pytest
from wavefront.simulate import make_wavefront, render
from wavefront.classical import demodulate
from wavefront.metrics import detrended_rmse


def test_classical_pipeline_low_noise():
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(5):
        phi, mask = make_wavefront(rng)
        _, u = demodulate(render(phi, rng, 0.03))
        errs.append(detrended_rmse(u, phi, mask))
    assert np.mean(errs) < 0.8


def test_unet_size_and_shape():
    torch = pytest.importorskip("torch")
    from wavefront.model import UNet
    m = UNet(2)
    assert 1.7e6 < sum(p.numel() for p in m.parameters()) < 2.1e6
    assert m(torch.zeros(2, 2, 128, 128)).shape == (2, 1, 128, 128)
