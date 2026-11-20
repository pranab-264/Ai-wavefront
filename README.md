# AI-Assisted Wavefront Reconstruction from a Single-Frame Interferogram

A hybrid physics + deep learning pipeline for single-frame phase retrieval.
A classical heterodyne Fourier-demodulation baseline produces a wrapped-phase estimate; a compact
two-channel U-Net (~1.9M parameters) takes the interferogram and that estimate and regresses the
**unwrapped** phase directly.

> **Scope:** simulation only (synthetic Zernike wavefronts). No experimental interferograms are used.
> The U-Net's advantage is a hypothesis until you run `evaluate.py` on the held-out test set.

## Pipeline
| Step | File |
|---|---|
| Forward model: `I = a + b·cos(2π(fx·x + fy·y) + φ) + n` with Zernike aberrations (defocus ±2.5, astigmatism/coma ±1.2, spherical ±1.5 rad), a Gaussian bump, noise σ 0.03–0.30, visibility 0.7–1.0, illumination gradient | `wavefront/simulate.py` |
| Classical baseline: carrier removal, Hann low-pass, angle, `skimage` unwrapping | `wavefront/classical.py` |
| U-Net (widths 32/64/128/256, 16×16 bottleneck, bilinear decoder, ×3 output scale) | `wavefront/model.py` |
| Metric: aperture RMSE after piston + tilt detrending | `wavefront/metrics.py` |
| Dataset: 160/40/60 split **by wavefront**; test wavefronts re-rendered at each σ | `wavefront/dataset.py` |

## Quick start
```bash
pip install -r requirements.txt
python make_dataset.py                 # data/{train,val,test}.npz
python train.py --inputs both          # hybrid model (AdamW, lr 1e-3, cosine, bs 16, 25 epochs)
python train.py --inputs interferogram # ablation: interferogram only
python train.py --inputs wrapped       # ablation: wrapped phase only
python evaluate.py                     # results/results.csv (mean, std, 95% CI, paired Wilcoxon)
python demo.py --sigma 0.1             # results/demo.png
pytest                                 # sanity tests
```
160 training samples is small. For a fairer test, try `python make_dataset.py --n_train 2000 --n_val 200`.

## Results
Classical baseline on 25 wavefronts per noise level (from the accompanying report):
RMSE 0.267 / 0.269 / 0.294 rad at σ = 0.03 / 0.10 / 0.30. Run `evaluate.py` to reproduce with this
code (its values will differ slightly because of different random seeds) and to measure the U-Net.
Do not quote U-Net numbers until they come from `results/results.csv`.

## Reproducibility notes
Carrier `(0.12, 0.08)` cycles/pixel and Hann cutoff 0.07 cycles/pixel are assumed known. Test data is
only touched by `evaluate.py`; checkpoints are chosen on validation RMSE. Seeds: train 1, val 2, test 3.

## Limitations / future work
Synthetic data, known carrier, limited aberration types. Next: real interferograms, carrier-error
robustness, vortices and discontinuities, physics-informed losses.

## References
Takeda, Ina & Kobayashi (1982), JOSA 72; Itoh (1982), Appl. Opt. 21; Ronneberger et al. (2015), U-Net, MICCAI.

## License
MIT
