# Showroom People-Tracker

Multi-camera retail analytics from NVR video feeds:

1. **Unique customer check-ins**
2. **Group count** (how many groups customers arrived in)
3. **Age & gender** (per-visitor, aggregate reporting)
4. **Employee time tracking** (in/out timestamps + dwell inside vs outside)

Target **95%+ accuracy** where achievable, with a **self-improving, human-in-the-loop** training
loop. Full architecture + roadmap: `C:\Users\namem\.claude\plans\bubbly-prancing-origami.md`.

> **Status: Phase 0 + Phase 1 core (prototype).** What works today: config-driven camera/line setup,
> a single-camera **detect → track → directional count** pipeline, an event/DB schema, and an accuracy
> harness. Cross-camera fusion, employee gallery, demographics and the retraining loop come next.

## Layout
```
common/      config loader + shared dataclasses (Track, FrameTracks)
ingest/      video sources (file/RTSP/synthetic), NVR URL builder, motion gating
vision/      geometry, directional counter, detector/tracker, single-cam pipeline
logic/       domain events + (later) visitor/group/employee logic
fusion/      [Phase 1] cross-camera multi-target tracking (MTMC) -> global IDs
storage/     Postgres + pgvector schema and writer
evaluation/  accuracy harness (MAPE vs hand-counted ground truth)
tools/       zone/line config preview + editor
hitl/        [Phase 4] human-in-the-loop review + active learning
training/    [Phase 4] retraining + eval gate
configs/     camera + pipeline config (copy the .example.yaml)
tests/       unit tests for geometry + counting
```

## Install
GPU-free demo + tests only:
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements-core.txt
```
Full model-backed pipeline (YOLO etc.) — install PyTorch for your CUDA first (https://pytorch.org):
```powershell
pip install -r requirements.txt
```

## Quickstart

**1) GPU-free synthetic demo** — scripted people cross a line; proves ingest→count→events→overlay end to end with no model:
```powershell
python run.py --synthetic --out outputs/synthetic_demo.mp4
```

**2) Preview your counting lines on a real frame** (validate placement before running):
```powershell
python -m tools.zone_editor --config configs/cameras.yaml --camera cam_entry --out outputs/cam_entry_lines.jpg
```

**3) Real pipeline on a video file or NVR RTSP channel:**
```powershell
# copy + edit config first
copy configs\cameras.example.yaml configs\cameras.yaml
$env:NVR_PASSWORD = "..."     # referenced as ${NVR_PASSWORD} in the YAML
python run.py --config configs/cameras.yaml --camera cam_entry --out outputs/cam_entry.mp4
# or a local file:
python run.py --config configs/cameras.yaml --camera cam_entry --source path\to\clip.mp4
```

## Measure accuracy
Hand-count entries/exits for a clip, drop them in a ground-truth file (see
`data/ground_truth/sample.json`), then:
```powershell
python -m evaluation.harness --truth data/ground_truth/sample.json --events outputs/cam_entry.events.json
```
Reports per-direction error and MAPE against the **5% Phase-1 gate**.

## Database (optional)
```powershell
docker compose -f storage/docker-compose.yml up -d     # Postgres + pgvector
# then set storage.enabled: true and the dsn in configs/cameras.yaml
```

## Tests
```powershell
pytest -q
```

## Batch day pipeline + self-improving loop (dummy data, GPU-free)
```powershell
python run_day.py     # full batch: fusion -> sessions -> demographics -> grouping -> timesheets -> report
python run_train.py   # HITL loop: cold-start -> human labels -> retrain -> eval-gate -> improved score
```
`run_day.py` produces the four outputs split into **customer** and **employee** tabs and compares to
ground truth. `run_train.py` demonstrates the data engine: an empty employee gallery miscounts staff
(score 0.0), a human labels them, the trainer enrolls them, and the eval gate promotes the improved
model (score 1.0) — only human-verified labels are ever trained on.

## Switching to real models
The orchestration is model-agnostic; real backends plug in behind the existing interfaces (needs a
GPU + `pip install -r requirements.txt`):
- **Detection/tracking:** `vision/track_source.py:YoloTrackSource` (YOLO11x + BoT-SORT).
- **Body ReID:** `fusion/embedder.py:build_embedder({"embedder":"reid", ...})` (torchreid OSNet; point
  `reid_weights` at CLIP-ReID for the accuracy-first tier).
- **Age/gender:** `vision/demographics.py:build_demographics({"backend":"mivolo", ...})`.
- **Face:** InsightFace (SCRFD + ArcFace) for the employee gallery (Phase 2 enrollment).

These were **not run in this environment** (no GPU/footage). Run `run.py` against a real clip/RTSP to
validate the live detection path on your hardware.

## Privacy / security notes
- Prefer a **VPN/secure tunnel** to the NVR over a raw public port-forward.
- Faces + demographics are sensitive: encrypt at rest/in transit, keep retention short, store
  embeddings over raw imagery where possible, post in-store signage/consent. If operating in India,
  align with the **DPDP Act 2023**. Report demographics in aggregate.
