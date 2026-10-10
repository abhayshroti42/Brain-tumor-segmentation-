# Brain Tumor Segmentation (research prototype)

A local web app that runs a trained 3D U-Net on preprocessed brain MRI patches and
shows the predicted tumor regions in an interactive viewer.

> **This is a research prototype. It is not a medical device and does not provide a
> diagnosis. Do not use it for clinical decisions.**

- **Backend:** Python, FastAPI, TensorFlow/Keras, NumPy
- **Frontend:** React, Vite, Tailwind CSS (slices are drawn on plain HTML canvases)
- **Model:** `models/best_model.keras`, a 3D U-Net trained separately on Kaggle
  (not included in this repository)

## What it does

1. You upload a preprocessed `.npz` patch.
2. The backend validates it, runs the model once, and keeps the result in memory.
3. The viewer shows three slice views (one per array axis), each with the MRI on the
   left and the MRI plus predicted segmentation on the right.
4. You can change overlay opacity, hide individual tumor classes, switch the displayed
   channel, zoom in (up to 8x, centered on the predicted voxels), jump to the slices
   holding the most predicted voxels, scroll through slices with the mouse wheel, and
   download the predicted mask.
5. The interface keeps its notes in an About dialog (the **i** button). The full list of
   limitations is in this README.

| Class | Meaning | Overlay color |
|---|---|---|
| 0 | Background | none |
| 1 | NCR/NET (necrotic and non-enhancing tumor core) | red |
| 2 | Edema | yellow |
| 3 | Enhancing tumor | blue |

## Model

| Item | Value |
|---|---|
| Input | `(128, 128, 128, 4)`, order `(H, W, D, C)` |
| Output | `(128, 128, 128, 4)` softmax probabilities, then `argmax` |
| Parameters | about 1.46 million |
| Training framework | TensorFlow/Keras (loads with TensorFlow 2.21.0 / Keras 3.15) |

The uploaded `.npz` stores the image as `(C, H, W, D)`; the backend transposes it to
`(H, W, D, C)`. **No normalization or other preprocessing is applied**, because the
training patches were already preprocessed.

## Project structure

```
brain-tumor-segmentation/
  backend/
    main.py                  FastAPI app and endpoints
    inference.py             model loading (once, at startup) and prediction
    results_store.py         in-memory store, latest 5 results
    preprocessing/
      npz_loader.py          validation and channel reordering for .npz uploads
    inspect_model.py         stage-1 check: does the model load, are shapes right
    check_equivalence.py     compares the API loading path with a direct load
    smoke_test.py            automated checks against a running backend
    requirements.txt
  frontend/
    src/App.jsx              page, upload, controls
    src/SliceView.jsx        canvas slice viewer
    src/api.js               backend calls
  models/
    best_model.keras         NOT in Git (see .gitignore)
  README.md
  .gitignore
```

## Setup (Windows, PowerShell)

Requirements: a Python version supported by TensorFlow 2.21 (check TensorFlow's
install page), and Node.js 20 or newer. TensorFlow on native Windows runs on the CPU
only.

```powershell
# from the project root
python -m venv backend\.venv
backend\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt

cd frontend
npm install
cd ..
```

If PowerShell blocks the activation script, run this once and retry:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Model file

Place the model at `models\best_model.keras`. To use another location, set an
environment variable before starting the backend:

```powershell
$env:BTS_MODEL_PATH = "C:\path\to\best_model.keras"
```

The model file is excluded from Git. Do not commit it unless its size and licensing
allow that.

## Run

Terminal 1, backend:

```powershell
backend\.venv\Scripts\Activate.ps1
cd backend
uvicorn main:app --port 8000
```

Wait for `Model ready.` in the log (a few seconds).

Terminal 2, frontend:

```powershell
cd frontend
npm run dev
```

Open http://localhost:5173. The status box should say **Model ready**. The backend
only accepts browser requests from `http://localhost:5173` and
`http://127.0.0.1:5173`.

## Input format

Upload a `.npz` file (max 100 MB) containing:

- `image`: float array of shape `(4, 128, 128, 128)`, channels first, all values finite
- `mask` (optional, ignored by the app)

Anything else is rejected with a clear error message. Training-style patches from the
dataset described below have this format.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Model readiness, path, TensorFlow version, error if any |
| POST | `/predict/npz` | Upload a preprocessed `.npz`; returns an `id`, voxel counts per class and inference time |
| POST | `/predict/nifti` | Always returns **501** (raw MRI not supported, see Limitations) |
| GET | `/results/{id}/mask` | Predicted label mask, raw `uint8` bytes, shape 128x128x128 |
| GET | `/results/{id}/image/{0-3}` | One channel as raw `uint8` bytes, **scaled for display only** |
| GET | `/results/{id}/mask.npz` | Downloadable mask (`mask` array, patch space) |

Results are kept in memory (latest 5) and disappear when the server restarts. One
prediction runs at a time. Expect about 5 to 10 seconds per patch on a CPU.

## Tests and checks

With the backend running:

```powershell
python backend\smoke_test.py backend\samples\SOME_PATCH.npz
```

This checks health, six kinds of invalid upload, the disabled NIfTI endpoint, an
unknown result id, and (with a sample) a valid prediction and the result endpoints.
**It does not measure model accuracy.**

Other scripts:

```powershell
python backend\inspect_model.py models\best_model.keras      # model loads, shapes match
python backend\check_equivalence.py path\to\patch.npz        # API loading == direct load
```

The viewer controls (opacity, class toggles, channel selector, download) were checked
manually by the developer, not by automated tests.

## Limitations (please read)

### Not supported

- **Raw MRI (NIfTI) input is disabled.** The preprocessing used to build the training
  patches (skull stripping, normalization, cropping, resampling and patch extraction
  details) is not documented well enough to reproduce. Generic resampling and z-scoring
  would not reproduce it, and the app would produce misleading predictions without any
  warning. The endpoint is disabled instead.
- **Dice and IoU are not computed.** No ground-truth comparison is built into the app.
  Any score shown elsewhere in this README is not a score for a new scan.

### Unverified

- **The channel order is unverified.** The dataset creator describes the channels as
  T1, T1ce, T2 and FLAIR, but no code proving which modality is stored at each index
  was found. The interface therefore labels them **Channel 0 to Channel 3**. The model
  itself is unaffected, since it uses whatever order is stored in the files.
- **Anatomical plane names are not assigned.** The patches carry no orientation
  metadata, so views are labeled by array axis (D, W, H), not axial, coronal or
  sagittal.
- **Dataset provenance details are taken from the dataset's description page** and have
  not been independently checked.

### Accuracy

- **Segmentation accuracy has not been validated in this app.** Treat every prediction
  as unverified.
- **Reported training-notebook validation results** (251 validation patches): NCR/NET
  Dice 0.5858, edema Dice 0.7983, enhancing tumor Dice 0.7562, mean tumor Dice 0.7134,
  mean tumor IoU 0.6264. These come from the notebook's own validation split. They are
  not a guarantee of performance on new data, and certainly not of clinical
  performance.
- **Three-patch spot check (not validation).** The loading path was verified identical
  to a direct notebook-style load. On `BraTS-GLI-00136-000_patch_0` the model assigned a
  mean probability of 0.958 to enhancing tumor inside its true region, but it
  over-predicted enhancing tumor (22,489 voxels against 8,438 true) and under-predicted
  edema. On `BraTS-GLI-00772-000_patch_0`, a very small lesion (704 true tumor voxels),
  it predicted 20 enhancing voxels and nothing else. A third patch with no tumor
  in its ground truth (`BraTS-GLI-01327-000_patch_0`) produced 14 voxels of predicted
  edema, all on real tissue at the patch edge (all four channels non-zero). These are
  false positives, and why they appeared is unknown. Whether these patches were in the training or validation
  split is unknown. Per-patch accuracy has not been evaluated in this project.
- **Small lesions may be missed.** See the spot check above. The cause has not been
  investigated.

### Other

- The model was trained on tumor-focused 128x128x128 patches. It has not been tested on
  whole-brain volumes.
- **Patches are crops, not whole brains.** In the sample checked, brain tissue touches
  all six faces of the volume (31% to 76% of voxels on each face), so the anatomy is cut
  off at the patch boundary. The viewer shows the full patch; it cannot show more than
  the file contains. Predictions near patch borders may be less reliable (not tested).
- The downloaded mask is in patch space with no spatial metadata, so it cannot be
  overlaid on an original scan.
- The displayed MRI is rescaled to 0-255 for viewing only (1st to 99th percentile of
  non-zero voxels). The model input is never rescaled.
- Uploads are processed in memory on your machine and are not stored beyond the latest
  five results. There is no authentication, database or cloud deployment; the app is
  meant for local use only.
- Inference runs on the CPU, so it is slow compared with a GPU.

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| Page says "Cannot reach backend" | Backend not running, or not on port 8000. Start `uvicorn main:app --port 8000`. |
| `/health` returns `degraded` | The `error` field says why, usually a wrong model path. Check `BTS_MODEL_PATH`. |
| `Cannot find package '@tailwindcss/vite'` | Run `npm install` inside `frontend`. |
| Result "not found" (404) | The server was restarted or more than 5 newer results exist. Analyze again. |
| Activation script blocked | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

## Data and credits

The model was trained on a preprocessed patch dataset from Kaggle
(`prathamhanda10/brats-2024-preprocessed-training-patches`), described there as derived
from the BraTS glioma segmentation data. The dataset page asks users to cite the
original BraTS challenge papers. If you share results, cite the BraTS challenge and
check the data's license terms yourself.

## Possible future work

- Verify the channel order from the dataset creator's code or the creator directly.
- Evaluate the model per patch on the held-out validation files, including by lesion
  size, in the training notebook.
- Support raw NIfTI input only after the exact preprocessing pipeline is verified.
- Show Dice and IoU only when a genuine ground-truth mask is supplied.