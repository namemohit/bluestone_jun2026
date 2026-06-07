"""Clip sinks — where motion clips land.

LocalSink writes to a folder (dev / on-prem). GcsSink uploads to Google Cloud Storage
(the cloud landing zone the nightly batch reads from). Same interface, swap by config.
"""
from __future__ import annotations

import shutil
from pathlib import Path


class LocalSink:
    """Filesystem sink (testing / on-prem staging)."""

    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, local_path: str, name: str) -> str:
        dest = self.root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, dest)
        return name

    def list(self, prefix: str = "") -> list[str]:
        return sorted(p.name for p in self.root.glob("*") if p.is_file() and p.name.startswith(prefix))

    def get(self, name: str, dest: str) -> str:
        shutil.copyfile(self.root / name, dest)
        return dest

    def delete(self, name: str) -> None:
        try:
            (self.root / name).unlink()
        except FileNotFoundError:
            pass


class GcsSink:
    """Google Cloud Storage sink (lazy import; needs `google-cloud-storage` + creds)."""

    def __init__(self, bucket: str, prefix: str = ""):
        from google.cloud import storage  # lazy/optional dependency

        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket)
        self.prefix = prefix.rstrip("/")

    def _key(self, name: str) -> str:
        return f"{self.prefix}/{name}" if self.prefix else name

    def put(self, local_path: str, name: str) -> str:
        self._bucket.blob(self._key(name)).upload_from_filename(local_path)
        return name

    def list(self, prefix: str = "") -> list[str]:
        full = self._key(prefix)
        n = len(self.prefix) + 1 if self.prefix else 0
        return sorted(b.name[n:] for b in self._client.list_blobs(self._bucket, prefix=full))

    def get(self, name: str, dest: str) -> str:
        self._bucket.blob(self._key(name)).download_to_filename(dest)
        return dest

    def delete(self, name: str) -> None:
        try:
            self._bucket.blob(self._key(name)).delete()
        except Exception:
            pass


def make_sink(spec: str):
    """`spec` = 'local:/path' or 'gcs:bucket/prefix'."""
    if spec.startswith("gcs:"):
        rest = spec[4:]
        bucket, _, prefix = rest.partition("/")
        return GcsSink(bucket, prefix)
    if spec.startswith("local:"):
        return LocalSink(spec[6:])
    return LocalSink(spec)
