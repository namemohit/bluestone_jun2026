# Deploy: on-site edge agent → GCS → nightly Cloud Run Job

```
 NVR (RTSP)  ──►  edge agent (mini-PC)  ──►  GCS bucket  ──►  Cloud Run Job + L4 (nightly)  ──►  DB
   on-site         motion clips, upload      landing zone      reads clips, processes, deletes
```

## 1) Edge agent (on-site mini-PC / NUC)
Runs `edge/agent.py`: pulls each camera's RTSP locally, motion-gates, uploads clips to GCS.

```bash
pip install -r requirements.txt google-cloud-storage
export GOOGLE_APPLICATION_CREDENTIALS=/etc/showroom/sa.json   # a GCS-writer service account
python -m edge.agent --config configs/cameras.yaml --sink gcs:my-clips-bucket/showroom-42
```
Run it as a **systemd service** so it restarts on boot/crash (`/etc/systemd/system/showroom-agent.service`):
```ini
[Service]
ExecStart=/usr/bin/python3 -m edge.agent --config /opt/showroom/configs/cameras.yaml --sink gcs:my-clips-bucket/showroom-42
Restart=always
WorkingDirectory=/opt/showroom
[Install]
WantedBy=multi-user.target
```
> Only motion clips leave the store. Prefer a VPN/tunnel; the NVR is never exposed to the internet.

## 2) Build the batch image
```bash
gcloud builds submit --tag asia-south1-docker.pkg.dev/PROJECT/showroom/batch:latest -f cloud/Dockerfile .
```

## 3) Cloud Run **Job** with a GPU (L4), scale-to-zero
```bash
gcloud run jobs create showroom-batch \
  --image asia-south1-docker.pkg.dev/PROJECT/showroom/batch:latest \
  --region asia-south1 --gpu 1 --gpu-type nvidia-l4 \
  --cpu 4 --memory 16Gi --max-retries 1 --task-timeout 3600 \
  --set-env-vars CLIPS_BUCKET=my-clips-bucket,STORE_ID=showroom-42
```

## 4) Trigger it nightly (Cloud Scheduler)
```bash
gcloud scheduler jobs create http showroom-batch-nightly \
  --location asia-south1 --schedule "30 22 * * *" --time-zone "Asia/Kolkata" \
  --uri "https://asia-south1-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/PROJECT/jobs/showroom-batch:run" \
  --oauth-service-account-email runner@PROJECT.iam.gserviceaccount.com --http-method POST
```

Cost: the job runs ~30–60 min/night then scales to zero — ~15–30 GPU-hr/month per store. Process the
**whole day in one run** (and you can batch several stores per run) to amortize cold start. Bake model
weights into the image (see Dockerfile) so cold starts don't re-download them.

> For lowest raw cost, swap the Cloud Run Job for a **Spot L4/T4 VM** started by Cloud Scheduler and
> stopped on completion — cheaper per GPU-hour, but you own the start/stop + preemption handling.
