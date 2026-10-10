"""Validate and convert an uploaded preprocessed .npz into model input.
No normalization here: the data is already preprocessed."""
import io

import numpy as np

EXPECTED_CHW = (4, 128, 128, 128)


class InvalidInput(ValueError):
    """Raised for bad uploads; the API turns it into HTTP 400."""


def load_npz_image(data: bytes) -> np.ndarray:
    """bytes of .npz -> float32 array of shape (128, 128, 128, 4)."""
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as npz:
            if "image" not in npz.files:
                raise InvalidInput(f"Missing 'image' key. Found keys: {npz.files}")
            image = npz["image"]
    except InvalidInput:
        raise
    except Exception as exc:
        raise InvalidInput(f"Not a valid .npz file: {type(exc).__name__}: {exc}")

    if image.ndim != 4 or tuple(image.shape) != EXPECTED_CHW:
        raise InvalidInput(f"'image' must have shape {EXPECTED_CHW} (C,H,W,D), got {image.shape}")
    if not np.issubdtype(image.dtype, np.floating):
        raise InvalidInput(f"'image' must be a float array, got {image.dtype}")
    if not np.isfinite(image).all():
        raise InvalidInput("'image' contains NaN or infinite values")

    # (C,H,W,D) -> (H,W,D,C), the order the model expects
    return np.ascontiguousarray(np.transpose(image, (1, 2, 3, 0)), dtype=np.float32)