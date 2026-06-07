"""Hikvision NVR recorded-footage ingestion over ISAPI (own clean implementation).

Pulls *recorded* footage for a time window from an NVR reached via the showroom's port-forward
(public IP + per-NVR web port), using ISAPI + HTTP Digest auth:

  1. device_info()      GET  /ISAPI/System/deviceInfo            -> connection test
  2. search_segments()  POST /ISAPI/ContentMgmt/search           -> recorded segments for a channel/window
  3. download_segment() POST /ISAPI/ContentMgmt/download          -> save each segment (direct-GET fallback)

Channel mapping: camera Cnn -> logical channel nn -> ISAPI trackID = nn*100 + 1.
Times are given in store-local (IST) and normalized to UTC `...Z` for the search.

CLI:
  # connection test
  python -m ingest.nvr_isapi --config configs/nvr.json --test
  # pull a window
  python -m ingest.nvr_isapi --config configs/nvr.json --nvr NVR02 --camera C05 \
      --start 2026-06-03T10:00:00 --end 2026-06-03T22:00:00 --out data/footage
"""
from __future__ import annotations

import argparse
import json
import re
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape
from zoneinfo import ZoneInfo

import requests
from requests.auth import HTTPDigestAuth

DEFAULT_TZ = "Asia/Kolkata"


# ---- helpers --------------------------------------------------------------
def nvr_id_from_name(name: str, fallback_idx: int = 1) -> str:
    digits = "".join(c for c in str(name or "") if c.isdigit())
    n = int(digits) if digits else int(max(1, fallback_idx))
    return f"NVR{n:02d}"


def channel_from_camera(camera: str) -> int:
    digits = "".join(c for c in str(camera or "") if c.isdigit())
    if not digits:
        raise ValueError(f"bad camera id {camera!r} (expected C05 / 5)")
    ch = int(digits)
    if ch <= 0:
        raise ValueError("channel must be >= 1")
    return ch


def track_id(channel: int, stream: str = "main") -> int:
    return int(channel) * 100 + (1 if str(stream).lower() == "main" else 2)


def uri_size(uri: str) -> int | None:
    """Hikvision playbackURI carries the exact byte size as ...&size=NNN."""
    m = re.search(r"[?&]size=(\d+)", str(uri or ""))
    return int(m.group(1)) if m else None


def to_utc_z(ts: str, tz: str = DEFAULT_TZ) -> str:
    """ISO timestamp (naive = store-local tz) -> 'YYYY-MM-DDTHH:MM:SSZ' in UTC."""
    raw = str(ts or "").strip()
    if not raw:
        raise ValueError("empty timestamp")
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(tz))
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_search_time(ts: str) -> str:
    """Wall-clock -> 'YYYY-MM-DDTHH:MM:SSZ' WITHOUT timezone conversion.

    This NVR reads ISAPI search times as **device-local (IST)**, so we send the local wall
    clock as typed (the trailing Z is just a format token here, not real UTC).
    """
    raw = str(ts or "").strip()
    if not raw:
        raise ValueError("empty timestamp")
    if raw.endswith("Z"):
        raw = raw[:-1]
    dt = datetime.fromisoformat(raw).replace(tzinfo=None)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_search_xml(channel: int, start_utc_z: str, end_utc_z: str, *, position: int = 0,
                     max_results: int = 200, stream: str = "main") -> str:
    return (
        "<CMSearchDescription>"
        f"<searchID>{uuid.uuid4()}</searchID>"
        f"<trackList><trackID>{track_id(channel, stream)}</trackID></trackList>"
        "<timeSpanList><timeSpan>"
        f"<startTime>{start_utc_z}</startTime><endTime>{end_utc_z}</endTime>"
        "</timeSpan></timeSpanList>"
        f"<maxResults>{int(max(1, max_results))}</maxResults>"
        f"<searchResultPostion>{int(position)}</searchResultPostion>"
        "<metadataList><metadataDescriptor>//recordType.meta.std-cgi.com</metadataDescriptor></metadataList>"
        "</CMSearchDescription>"
    )


def parse_segments(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    segs: list[dict] = []
    for m in root.findall(".//{*}searchMatchItem"):
        s = m.find(".//{*}startTime")
        e = m.find(".//{*}endTime")
        u = m.find(".//{*}playbackURI")
        if s is None or e is None:
            continue
        segs.append({
            "start_time": (s.text or "").strip(),
            "end_time": (e.text or "").strip(),
            "playback_uri": (u.text or "").strip() if u is not None else "",
        })
    return segs


# ---- config ---------------------------------------------------------------
@dataclass
class NvrDevice:
    nvr_id: str
    name: str
    host: str
    port: int
    username: str
    password: str
    channels: list[int] = field(default_factory=list)


def load_devices(cfg: dict) -> dict[str, NvrDevice]:
    use_public = bool(cfg.get("use_public_ip", True))
    public_ip = str(cfg.get("nvr_public_ip", "")).strip()
    out: dict[str, NvrDevice] = {}
    for idx, d in enumerate(cfg.get("devices", []), start=1):
        nid = nvr_id_from_name(d.get("name", ""), idx)
        host = public_ip if use_public else str(d.get("internal_ip", ""))
        port = int(d.get("public_port") if use_public else d.get("internal_port"))
        out[nid] = NvrDevice(
            nvr_id=nid, name=str(d.get("name", "")), host=host, port=port,
            username=str(d.get("username", "")), password=str(d.get("password", "")),
            channels=[int(c) for c in d.get("channels", []) if str(c).strip()],
        )
    return out


def load_config(path: str) -> dict[str, NvrDevice]:
    return load_devices(json.loads(Path(path).read_text(encoding="utf-8")))


# ---- client ---------------------------------------------------------------
class IsapiFootageClient:
    def __init__(self, device: NvrDevice, *, session=None, timeout: float = 120.0,
                 max_bytes: int | None = None):
        self.device = device
        self.session = session or requests.Session()
        self.auth = HTTPDigestAuth(device.username, device.password)
        self.timeout = timeout
        self.max_bytes = max_bytes  # hard cap so an untrimmed segment can't run away

    @property
    def base_url(self) -> str:
        return f"http://{self.device.host}:{self.device.port}"

    def device_info(self) -> dict:
        try:
            r = self.session.get(f"{self.base_url}/ISAPI/System/deviceInfo",
                                 auth=self.auth, timeout=self.timeout)
        except Exception as e:
            return {"ok": False, "error": str(e)}
        if r.status_code != 200:
            return {"ok": False, "status": r.status_code}
        info = {"ok": True, "status": 200}
        try:
            root = ET.fromstring(r.text)
            for tag in ("deviceName", "model", "serialNumber"):
                node = root.find(f".//{{*}}{tag}")
                if node is not None:
                    info[tag] = (node.text or "").strip()
        except Exception:
            pass
        return info

    def search_segments(self, channel: int, start_utc_z: str, end_utc_z: str,
                        *, max_results: int = 200, stream: str = "main") -> list[dict]:
        url = f"{self.base_url}/ISAPI/ContentMgmt/search"
        position, seen, out = 0, set(), []
        while True:
            xml = build_search_xml(channel, start_utc_z, end_utc_z,
                                   position=position, max_results=max_results, stream=stream)
            r = self.session.post(url, auth=self.auth, data=xml.encode("utf-8"),
                                  headers={"Content-Type": "application/xml"}, timeout=self.timeout)
            if r.status_code != 200:
                raise RuntimeError(f"search HTTP {r.status_code}: {r.text[:300]!r}")
            page = parse_segments(r.text)
            if not page:
                break
            fresh = 0
            for seg in page:
                key = (seg["start_time"], seg["end_time"], seg["playback_uri"])
                if key in seen:
                    continue
                seen.add(key)
                out.append(seg)
                fresh += 1
            if fresh < max_results:
                break
            position += fresh
        out.sort(key=lambda s: (s["start_time"], s["end_time"]))
        for i, s in enumerate(out, start=1):
            s["segment_number"] = i
        return out

    def _stream_to_file(self, resp, out_file: Path, expected: int | None = None) -> int:
        ct = (resp.headers.get("Content-Type") or "").lower() if getattr(resp, "headers", None) else ""
        tmp = out_file.with_suffix(out_file.suffix + ".part")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        total, first = 0, True
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=512 * 1024):
                if not chunk:
                    continue
                if first:
                    first = False
                    if any(x in ct for x in ("xml", "json", "text", "html")):
                        raise RuntimeError(f"non-binary response (content-type={ct!r})")
                fh.write(chunk)
                total += len(chunk)
                if expected and total >= expected:
                    break  # NVR may not signal EOF; stop at the known segment size
                if self.max_bytes and total >= self.max_bytes:
                    break  # hard cap (segments stream untrimmed; don't run away)
        try:
            resp.close()
        except Exception:
            pass
        if total <= 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError("empty download")
        tmp.replace(out_file)
        return total

    def download_segment(self, playback_uri: str, out_file: Path) -> int:
        expected = uri_size(playback_uri)
        url = f"{self.base_url}/ISAPI/ContentMgmt/download"
        body = f"<downloadRequest><playbackURI>{xml_escape(playback_uri)}</playbackURI></downloadRequest>"
        r = self.session.post(url, auth=self.auth, data=body.encode("utf-8"),
                              headers={"Content-Type": "application/xml"}, stream=True, timeout=self.timeout)
        if getattr(r, "status_code", 0) in (200, 206):
            return self._stream_to_file(r, out_file, expected)
        # fallback: some firmwares serve the playbackURI directly
        r2 = self.session.get(playback_uri, auth=self.auth, stream=True, timeout=self.timeout)
        if getattr(r2, "status_code", 0) not in (200, 206):
            raise RuntimeError(f"download failed (POST {getattr(r,'status_code',None)}, GET {getattr(r2,'status_code',None)})")
        return self._stream_to_file(r2, out_file, expected)

    def download_window(self, channel: int, start: str, end: str, out_dir: str,
                        *, tz: str = DEFAULT_TZ, stream: str = "main") -> dict:
        start_z, end_z = to_search_time(start), to_search_time(end)
        segs = self.search_segments(channel, start_z, end_z, stream=stream)
        out_dir = Path(out_dir) / f"{self.device.nvr_id}_C{channel:02d}"
        results = []
        for seg in segs:
            name = f"seg_{seg['segment_number']:04d}.mp4"
            f = out_dir / name
            try:
                size = self.download_segment(seg["playback_uri"], f)
                results.append({**seg, "file": str(f), "size_bytes": size, "status": "ok"})
            except Exception as e:
                results.append({**seg, "status": "failed", "error": str(e)})
        return {"channel": channel, "start_utc": start_z, "end_utc": end_z,
                "segments": len(segs), "results": results}


# ---- CLI ------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Hikvision NVR recorded-footage ingestion (ISAPI)")
    ap.add_argument("--config", required=True, help="NVR config JSON (see configs/nvr.example.json)")
    ap.add_argument("--test", action="store_true", help="connection test (deviceInfo) for all NVRs")
    ap.add_argument("--list", action="store_true", help="list recorded segments only (no download)")
    ap.add_argument("--nvr", help="NVR id, e.g. NVR02")
    ap.add_argument("--camera", help="camera id, e.g. C05")
    ap.add_argument("--start", help="store-local start, e.g. 2026-06-03T10:00:00")
    ap.add_argument("--end", help="store-local end, e.g. 2026-06-03T22:00:00")
    ap.add_argument("--out", default="data/footage")
    ap.add_argument("--tz", default=DEFAULT_TZ)
    ap.add_argument("--stream", default="main", choices=["main", "sub"],
                    help="main (full-res) or sub (low-bitrate, for slow uplinks)")
    ap.add_argument("--max-mb", type=float, default=0, help="cap download size in MB (0 = no cap)")
    args = ap.parse_args()

    devices = load_config(args.config)

    if args.test:
        for nid, dev in devices.items():
            info = IsapiFootageClient(dev).device_info()
            tag = "OK" if info.get("ok") else "FAIL"
            print(f"[{tag}] {nid} {dev.host}:{dev.port}  {info.get('deviceName','')} {info.get('model','')} "
                  f"{info.get('error', info.get('status',''))}")
        return

    for req in ("nvr", "camera", "start", "end"):
        if not getattr(args, req):
            ap.error(f"--{req} required (or use --test)")
    dev = devices.get(args.nvr)
    if dev is None:
        ap.error(f"NVR {args.nvr} not in config. Known: {', '.join(devices)}")
    ch = channel_from_camera(args.camera)
    client = IsapiFootageClient(dev, max_bytes=int(args.max_mb * 1024 * 1024) if args.max_mb else None)
    if args.list:
        start_z, end_z = to_search_time(args.start), to_search_time(args.end)
        segs = client.search_segments(ch, start_z, end_z, stream=args.stream)
        print(json.dumps({"nvr": args.nvr, "camera": args.camera, "channel": ch, "stream": args.stream,
                          "start_utc": start_z, "end_utc": end_z,
                          "segment_count": len(segs), "segments": segs}, indent=2))
        return
    summary = client.download_window(ch, args.start, args.end, args.out, tz=args.tz, stream=args.stream)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
