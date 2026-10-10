"""Model loading and readiness state. Prediction logic comes in Stage 3."""
import os
from pathlib import Path
import threading
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.keras"

INPUT_SHAPE = (128, 128, 128, 4)   # (H, W, D, C)
CLASS_NAMES = {0: "Background", 1: "NCR/NET", 2: "Edema", 3: "Enhancing Tumor"}


def get_model_path() -> Path:
    """Use BTS_MODEL_PATH if set, otherwise models/best_model.keras."""
    return Path(os.environ.get("BTS_MODEL_PATH", DEFAULT_MODEL_PATH))


class ModelService:
    def __init__(self) -> None:
        self.model = None
        self.error: str | None = None
        self.path: Path = get_model_path()
        self._lock = threading.Lock()  # one prediction at a time

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        """Load the model once. Failures are stored, not raised, so /health can report them."""
        self.path = get_model_path()
        if not self.path.is_file():
            self.error = f"Model file not found: {self.path}"
            return
        try:
            model = tf.keras.models.load_model(self.path, compile=False)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return

        if tuple(model.input_shape[1:]) != INPUT_SHAPE:
            self.error = f"Unexpected input shape {model.input_shape}"
            return
        if tuple(model.output_shape[1:]) != INPUT_SHAPE:
            self.error = f"Unexpected output shape {model.output_shape}"
            return

        self.model = model
        self.error = None

    def predict(self, volume: np.ndarray) -> np.ndarray:
        """volume (128,128,128,4) float32 -> label mask (128,128,128) uint8."""
        if not self.ready:
            raise RuntimeError("Model is not loaded")
        with self._lock:
            probs = self.model.predict(volume[np.newaxis], verbose=0)[0]
        return np.argmax(probs, axis=-1).astype(np.uint8)
    def info(self) -> dict:
        return {
            "model_ready": self.ready,
            "model_path": str(self.path),
            "error": self.error,
            "input_shape": list(INPUT_SHAPE),
            "classes": CLASS_NAMES,
            "tensorflow_version": tf.__version__,
        }


model_service = ModelService()