# Deploy the stack to Cloud Run (GPU) — project `ogretailos-474407`

On-demand, scale-to-zero, pay-per-run. Footage lives in GCS; the Cloud Run **Job** (NVIDIA L4)
reads a day's clips, runs L1–L5, writes the report + crops back to GCS.

```
[bulk download / edge agent] → GCS → [Cloud Run GPU Job: batch.run_stack] → GCS (report + crops)
        (network, not Cloud Run)      (pay only while running; free credits apply)
```

Set once:
```bash
PROJECT=ogretailos-474407
REGION=asia-south1          # Mumbai (closest); use europe-west1/us-central1 if L4 quota is there
BUCKET=gs://bluestone-showroom
gcloud config set project $PROJECT
```

## 1) Enable APIs
```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  cloudbuild.googleapis.com storage.googleapis.com   # storage already on
```

## 2) GPU quota (the one likely gate)
Cloud Run L4 needs quota: **"Total Nvidia L4 GPU allocation, per project per region"** for Cloud Run.
New projects start at **0** → request an increase (Console → IAM & Admin → Quotas → filter "Cloud Run" + "L4").
Approval is often minutes–hours. *(Vertex AI Workbench can sidestep this if quota stalls.)*

## 3) GCS bucket + get footage in
```bash
gsutil mb -l $REGION $BUCKET
# Test now with the local 6 PM clips we already have:
gsutil -m cp data/footage_rtsp/c05_6pm_ist.ts $BUCKET/s14/2026-06-03/c05.ts
gsutil -m cp data/footage_rtsp/c11_6pm_ist.ts $BUCKET/s14/2026-06-03/c11.ts
gsutil -m cp data/footage_rtsp/c14_6pm_ist.ts $BUCKET/s14/2026-06-03/c14.ts
# Full day later: bulk-ISAPI-download → same gs://.../<store>/<date>/<cam>.ts layout.
```

## 4) Build the image (Cloud Build → Artifact Registry)
```bash
gcloud artifacts repositories create showroom --repository-format=docker --location=$REGION
gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT/showroom/stack:latest -f cloud/Dockerfile .
```

## 5) Create the Cloud Run GPU Job
```bash
gcloud run jobs create showroom-stack \
  --image $REGION-docker.pkg.dev/$PROJECT/showroom/stack:latest \
  --region $REGION --gpu 1 --gpu-type nvidia-l4 \
  --cpu 8 --memory 32Gi --max-retries 1 --task-timeout 21600 \
  --set-env-vars CLIPS_BUCKET=bluestone-showroom,STORE=s14,DATE=2026-06-03,MANIFEST=configs/store_manifest.example.json
```

## 6) Run it (for a given day)
```bash
gcloud run jobs execute showroom-stack --region $REGION \
  --update-env-vars DATE=2026-06-03
# outputs land at gs://bluestone-showroom/s14/2026-06-03/outputs/ (visits.csv, occupancy.csv, crops/)
```

## Local dry-run (no GCS, uses clips on disk) — sanity-check before cloud
```bash
python -m batch.run_stack --local data/footage_rtsp --store s14 --date 2026-06-03 \
    --manifest configs/store_manifest.example.json --out outputs/day_report.json
```

## Cost
L4 ≈ a few $/full-day run, billed only while running, then scales to zero. Weights are baked into
the image (no cold-start re-download). Free credits apply to Cloud Run + GCS + Build.
```
```
