"""python make_dataset.py [--n_train 160 --n_val 40 --n_test 60]"""
import argparse, os
import numpy as np
from wavefront.dataset import build_train_like, build_test

ap = argparse.ArgumentParser()
ap.add_argument("--n_train", type=int, default=160)
ap.add_argument("--n_val", type=int, default=40)
ap.add_argument("--n_test", type=int, default=60)
a = ap.parse_args()
os.makedirs("data", exist_ok=True)
np.savez_compressed("data/train.npz", **build_train_like(a.n_train, seed=1))
np.savez_compressed("data/val.npz", **build_train_like(a.n_val, seed=2))
np.savez_compressed("data/test.npz", **build_test(a.n_test, seed=3))  # touch only for final evaluation
print("saved data/{train,val,test}.npz")
