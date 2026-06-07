# Cloud Run Job image (GPU: NVIDIA L4) for the layered showroom stack (L1-L5 + OSNet ReID).
# Base bundles CUDA + PyTorch so YOLO11 + OSNet run on the GPU.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY cloud/requirements.txt ./req.txt
RUN pip install --no-cache-dir -r req.txt

COPY . .

# Bake model weights so scale-to-zero cold starts don't re-download (YOLO ~40MB, OSNet ~11MB).
RUN python -c "from ultralytics import YOLO; YOLO('yolo11m.pt')" || true \
 && python -c "import torch; from boxmot.reid.core.reid import ReID; ReID(weights='osnet_x1_0_msmt17.pt', device='cpu')" || true

ENV PYTHONUNBUFFERED=1
# Cloud Run Job sets: STORE, DATE, CLIPS_BUCKET, MANIFEST. Clips read from
# gs://$CLIPS_BUCKET/$STORE/$DATE/<cam>.ts ; outputs written to .../outputs/.
ENTRYPOINT ["sh","-c","python -m batch.run_stack --manifest ${MANIFEST:-configs/store_manifest.example.json} --store $STORE --date $DATE --bucket $CLIPS_BUCKET --out /tmp/day_report.json"]
