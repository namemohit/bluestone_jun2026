"""Read-only public dashboard for Cloud Run — friends watch the live results.

Slim on purpose: mounts ONLY the HITL read endpoints (no synthetic DashboardState, no cv2/torch).
With HITL_READONLY=1 the label/undo/link endpoints refuse writes, so the public view can't change
your data — only the local console edits. Reads the showroom Supabase schema over psycopg2 and
proxies crop images from the private Storage bucket. Creds come from Cloud Run env vars
(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, DB_URL).

  uvicorn dashboard.cloud_app:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

import pathlib

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from dashboard.hitl_api import router as hitl_router

STATIC = pathlib.Path(__file__).parent / "static"
app = FastAPI(title="BlueStone Showroom — Live Results")
app.include_router(hitl_router)


@app.get("/", response_class=HTMLResponse)
@app.get("/review", response_class=HTMLResponse)
def review():
    return (STATIC / "review.html").read_text(encoding="utf-8")


@app.get("/healthz")
def healthz():
    return {"ok": True}
