"""FastAPI entry point. Loads the model once at startup."""
from fastapi import FastAPI, File, HTTPException, UploadFile
import logging
from contextlib import asynccontextmanager
import io
from fastapi.responses import Response
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from inference import model_service
import time

import numpy as np

import results_store
from preprocessing.npz_loader import InvalidInput, load_npz_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bts")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model from %s ...", model_service.path)
    model_service.load()
    if model_service.ready:
        logger.info("Model ready.")
    else:
        logger.error("Model failed to load: %s", model_service.error)
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Brain Tumor Segmentation API", lifespan=lifespan)

# Allow the Vite dev server (used in a later stage) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok" if model_service.ready else "degraded", **model_service.info()}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB


@app.post("/predict/npz")
def predict_npz(file: UploadFile = File(...)):
    if not model_service.ready:
        raise HTTPException(503, f"Model not ready: {model_service.error}")
    if not (file.filename or "").lower().endswith(".npz"):
        raise HTTPException(400, "Please upload a .npz file")

    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (limit 100 MB)")

    try:
        volume = load_npz_image(data)
    except InvalidInput as exc:
        raise HTTPException(400, str(exc))

    start = time.time()
    mask = model_service.predict(volume)
    elapsed = time.time() - start

    counts = np.bincount(mask.ravel(), minlength=4)
    return {
        "id": results_store.save(volume,mask),
        "mask_shape": list(mask.shape),
        "voxel_counts": {name: int(counts[i]) for i, name in model_service_names().items()},
        "inference_seconds": round(elapsed, 2),
        "note": "Prediction only. No Dice/IoU: no ground-truth mask was evaluated.",
    }


def model_service_names():
    from inference import CLASS_NAMES
    return CLASS_NAMES
@app.post("/predict/nifti")
def predict_nifti():
    raise HTTPException(
        501,
        "Raw MRI (NIfTI) inference is not supported: the preprocessing used to "
        "build the training patches (skull stripping, normalization, tumor-focused "
        "cropping) has not been verified. Use /predict/npz with preprocessed data.",
    )
CHANNEL_NAMES = ["T1", "T1ce", "T2", "FLAIR"]
_BIN = "application/octet-stream"


def _get_result(rid: str) -> dict:
    result = results_store.get(rid)
    if result is None:
        raise HTTPException(404, "Result not found (only the latest 5 are kept in memory)")
    return result


@app.get("/results/{rid}/mask")
def result_mask(rid: str):
    """Label mask as raw uint8 bytes, shape (128,128,128), C order."""
    mask = _get_result(rid)["mask"]
    return Response(mask.tobytes(), media_type=_BIN, headers={"X-Shape": "128,128,128"})


@app.get("/results/{rid}/image/{channel}")
def result_image(rid: str, channel: int):
    """One modality as raw uint8 bytes for DISPLAY only (not model input)."""
    if channel not in range(4):
        raise HTTPException(400, f"channel must be 0-3 ({CHANNEL_NAMES})")
    vol = _get_result(rid)["volume"][..., channel]
    brain = vol[vol != 0]
    lo, hi = np.percentile(brain, [1, 99]) if brain.size else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1.0
    img = (np.clip((vol - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    img[vol == 0] = 0  # skull-stripped background renders pure black
    return Response(img.tobytes(), media_type=_BIN, headers={"X-Shape": "128,128,128"})


@app.get("/results/{rid}/mask.npz")
def download_mask(rid: str):
    """Downloadable segmentation (patch space, no spatial metadata)."""
    buf = io.BytesIO()
    np.savez_compressed(buf, mask=_get_result(rid)["mask"])
    return Response(
        buf.getvalue(),
        media_type=_BIN,
        headers={"Content-Disposition": 'attachment; filename="segmentation.npz"'},
    )