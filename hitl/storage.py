"""Supabase Storage (REST) for crop thumbnails — so the Cloud Run dashboard (no local disk) can
show them. Uses `requests`, NOT supabase-py (the local supabase/ dir shadows that import).

The bucket is PRIVATE (customer faces) — the dashboard proxies images via its own /crop endpoint,
so a raw Storage URL is never exposed to the browser. The storage key mirrors the local path with
the `outputs/` prefix stripped, so it's derivable from the crop path already stored in the DB:
  outputs/2026-06-03_1122/L1_C11/crops/trk_0039.jpg  ->  2026-06-03_1122/L1_C11/crops/trk_0039.jpg
"""
from __future__ import annotations

import os
import pathlib

import requests

_CACHE: dict | None = None


def _cfg() -> tuple[str, str, str]:
    global _CACHE
    if _CACHE is None:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass
        _CACHE = (os.environ.get("SUPABASE_URL", ""), os.environ.get("SUPABASE_SERVICE_ROLE_KEY", ""),
                  os.environ.get("SHOWROOM_BUCKET", "showroom-crops"))
    return _CACHE


def _headers() -> dict:
    _, key, _ = _cfg()
    return {"Authorization": f"Bearer {key}", "apikey": key}


def configured() -> bool:
    url, key, _ = _cfg()
    return bool(url and key)


def key_for(local_path: str) -> str:
    p = str(local_path).replace("\\", "/")
    return p[len("outputs/"):] if p.startswith("outputs/") else p.lstrip("/")


def upload_crop(local_path: str) -> str | None:
    """Upload one crop; returns its storage key (or None if missing/unconfigured)."""
    if not configured() or not os.path.exists(local_path):
        return None
    url, _, bucket = _cfg()
    k = key_for(local_path)
    try:
        requests.post(f"{url}/storage/v1/object/{bucket}/{k}",
                      headers={**_headers(), "Content-Type": "image/jpeg", "x-upsert": "true"},
                      data=pathlib.Path(local_path).read_bytes(), timeout=30)
    except Exception:
        return None
    return k


def fetch_crop(storage_key: str) -> bytes | None:
    """Fetch one crop's bytes from the private bucket (dashboard proxies this to the browser)."""
    if not configured():
        return None
    url, _, bucket = _cfg()
    try:
        r = requests.get(f"{url}/storage/v1/object/authenticated/{bucket}/{storage_key}",
                         headers=_headers(), timeout=30)
        return r.content if r.status_code == 200 else None
    except Exception:
        return None
