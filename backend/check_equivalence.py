"""Compare the API's loading path with a direct notebook-style path. Changes nothing."""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from preprocessing.npz_loader import load_npz_image

path = Path(sys.argv[1])
model_path = Path(__file__).resolve().parent.parent / "models" / "best_model.keras"
model = tf.keras.models.load_model(model_path, compile=False)

# Notebook-style path: transpose (C,H,W,D) -> (H,W,D,C), nothing else
data = np.load(path)
direct_in = np.transpose(data["image"], (1, 2, 3, 0)).astype(np.float32)
gt = data["mask"]

# Backend path
api_in = load_npz_image(path.read_bytes())

print("inputs identical:", np.array_equal(direct_in, api_in))
probs = model.predict(api_in[None], verbose=0)[0]
pred = np.argmax(probs, axis=-1)
probs2 = model.predict(direct_in[None], verbose=0)[0]
print("predicted masks identical:", np.array_equal(pred, np.argmax(probs2, axis=-1)))
print("GT counts  ", np.bincount(gt.ravel(), minlength=4))
print("Pred counts", np.bincount(pred.ravel(), minlength=4))

for k in (1, 2, 3):
    region = gt == k
    if region.any():
        print(f"class {k}: mean prob of class {k} inside its GT region = {probs[..., k][region].mean():.3f}")