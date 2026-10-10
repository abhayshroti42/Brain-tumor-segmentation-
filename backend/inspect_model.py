"""Stage 1: confirm best_model.keras loads and has the expected shapes."""
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

EXPECTED_INPUT = (None, 128, 128, 128, 4)
EXPECTED_OUTPUT = (None, 128, 128, 128, 4)


def main() -> int:
    model_path = Path(sys.argv[1] if len(sys.argv) > 1 else "models/best_model.keras")
    print(f"TensorFlow: {tf.__version__}")
    print(f"Keras:      {tf.keras.__version__}")
    print(f"Model path: {model_path.resolve()}")

    if not model_path.is_file():
        print("ERROR: model file not found.")
        return 1
    print(f"File size:  {model_path.stat().st_size / 1e6:.1f} MB")

    # compile=False: inference only, avoids needing custom loss/metric objects
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as exc:
        print(f"ERROR loading model: {type(exc).__name__}: {exc}")
        return 1
    print("Model loaded.")

    in_shape = tuple(model.input_shape)
    out_shape = tuple(model.output_shape)
    print(f"Input shape:  {in_shape}  (expected {EXPECTED_INPUT})")
    print(f"Output shape: {out_shape}  (expected {EXPECTED_OUTPUT})")
    print(f"Parameters:   {model.count_params():,}")

    last = model.layers[-1]
    act = getattr(getattr(last, "activation", None), "__name__", "unknown")
    print(f"Last layer:   {last.name} ({type(last).__name__}), activation={act}")

    ok = in_shape == EXPECTED_INPUT and out_shape == EXPECTED_OUTPUT

    # Smoke test: one forward pass on zeros (checks the graph runs, NOT accuracy)
    print("Running one forward pass on a zero volume (may take a while on CPU)...")
    start = time.time()
    pred = model.predict(np.zeros((1, 128, 128, 128, 4), np.float32), verbose=0)
    print(f"Forward pass: {time.time() - start:.1f} s, output shape {pred.shape}")
    print(f"Prob sum per voxel: min={pred.sum(-1).min():.4f} max={pred.sum(-1).max():.4f}")

    print("RESULT:", "PASS" if ok else "CHECK SHAPES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())