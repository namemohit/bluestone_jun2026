"""Review-console backend (FastAPI) over the dummy day.

Control plane for the HITL loop: holds the live model params, a LabelStore, and a model
registry. Endpoints power four tabs — Overview, Review (affirm/correct cards), Pipeline (live
per-step params), Training (cadence/target/auto-promote + registry).

  uvicorn dashboard.app:app --port 8000
"""
from __future__ import annotations

import pathlib
from dataclasses import replace

import yaml
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from hitl.corrections import Correction, LabelStore
from ingest.nvr import build_rtsp_url, masked
from hitl.loop import cold_start_params, collect_corrections
from hitl.review import build_review_cards
from run_day import run_pipeline
from training.metrics import day_score
from training.registry import ModelRegistry
from training.trainer import train_candidate

STATIC = pathlib.Path(__file__).parent / "static"
CONFIG_PATH = pathlib.Path(__file__).parent.parent / "configs" / "cameras.yaml"


def default_sources() -> dict:
    return {
        "nvr": {"brand": "hikvision", "host": "", "rtsp_port": 554, "username": "admin", "password": ""},
        "cameras": [
            {"id": "cam_entry", "role": "entry", "channel": 1, "stream": "main"},
            {"id": "cam_exit", "role": "exit", "channel": 2, "stream": "main"},
            {"id": "cam_common", "role": "common", "channel": 3, "stream": "main"},
        ],
    }


def load_sources() -> dict:
    if CONFIG_PATH.exists():
        try:
            raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
            nvr = raw.get("nvr") or default_sources()["nvr"]
            cams = [{"id": c.get("id"), "role": c.get("role", "common"),
                     "channel": c.get("channel", 1), "stream": c.get("stream", "main")}
                    for c in (raw.get("cameras") or [])]
            return {"nvr": nvr, "cameras": cams or default_sources()["cameras"]}
        except Exception:
            pass
    return default_sources()


def write_cameras_yaml(sources: dict) -> None:
    cfg = {
        "site": {"name": "BlueStone Showroom", "timezone": "Asia/Kolkata"},
        "nvr": sources["nvr"],
        "cameras": [{**c, "source": None, "fps_sample": 8, "counting_lines": [], "rois": []}
                    for c in sources["cameras"]],
        "detector": {"backend": "yolo", "weights": "yolo11x.pt", "conf": 0.35, "iou": 0.5,
                     "classes": [0], "device": None, "imgsz": 960, "tracker": "botsort.yaml"},
        "sessions": {"window_sec": 7200, "match_threshold": 0.8, "review_threshold": 0.6,
                     "employee_threshold": 0.7, "embedder": "reid"},
        "storage": {"enabled": False, "dsn": "postgresql://tracker:tracker@localhost:5432/showroom"},
    }
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def sources_preview(sources: dict) -> list:
    out = []
    for cam in sources["cameras"]:
        try:
            out.append({"id": cam["id"], "role": cam.get("role"),
                        "url": masked(build_rtsp_url(sources["nvr"], cam))})
        except Exception as e:
            out.append({"id": cam["id"], "role": cam.get("role"), "url": "", "error": str(e)})
    return out


def probe_camera(nvr: dict, cam: dict, timeout_ms: int = 4000) -> dict:
    """Best-effort RTSP reachability check (works on a machine with NVR/tunnel access)."""
    import cv2

    try:
        url = build_rtsp_url(nvr, cam)
    except Exception as e:
        return {"id": cam["id"], "ok": False, "error": str(e)}
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
    except Exception:
        pass
    opened = cap.isOpened()
    frame_ok = False
    if opened:
        try:
            r, f = cap.read()
            frame_ok = bool(r) and f is not None
        except Exception:
            frame_ok = False
    cap.release()
    return {"id": cam["id"], "ok": bool(frame_ok),
            "error": "" if frame_ok else ("opened, no frame" if opened else "could not connect")}


class DashboardState:
    def __init__(self):
        self.sources = load_sources()  # NVR + cameras config (persists across reset)
        self.reset()

    # ---- sources (camera / NVR config) ---------------------------------
    def sources_view(self) -> dict:
        return {"nvr": self.sources["nvr"], "cameras": self.sources["cameras"],
                "preview": sources_preview(self.sources), "config_path": str(CONFIG_PATH)}

    def save_sources(self, data: dict) -> dict:
        nvr = {**self.sources["nvr"], **(data.get("nvr") or {})}
        cams = data.get("cameras") or self.sources["cameras"]
        self.sources = {"nvr": nvr, "cameras": cams}
        write_cameras_yaml(self.sources)
        return self.sources_view()

    def test_sources(self) -> list:
        return [probe_camera(self.sources["nvr"], c) for c in self.sources["cameras"]]

    def reset(self):
        self.params = cold_start_params()
        self.store = LabelStore()
        self.registry = ModelRegistry()
        self.applied_ids: set[str] = set()
        self._pending: dict[str, Correction] = {}
        self._labels_at_last_train = 0
        self.last_result: dict | None = None
        self.employees: dict[int, str] = {}  # employee_id -> name (filled in by the reviewer)
        self.pipeline_config = {
            "detect_conf": 0.35, "detect_iou": 0.5, "detect_model": "yolo11x",
            "reid_sim": 0.6, "reid_model": "clip-reid",
        }
        self.train_settings = {"cadence": "nightly", "min_new_labels": 5,
                               "target": 0.95, "auto_promote": True}
        report, gt, _ = run_pipeline(params=self.params)
        score, checks = day_score(report, gt)
        mv = self.registry.register(self.params, {"score": score, "checks": checks})
        self.registry.promote(mv.version)

    # ---- views ----------------------------------------------------------
    def params_view(self) -> list:
        p, cfg = self.params, self.pipeline_config
        return [
            {"step": "1 · Detect", "desc": "YOLO person detector — applies to real footage", "fields": [
                {"key": "detect.confidence", "label": "Confidence", "value": cfg["detect_conf"],
                 "min": 0.1, "max": 0.9, "step": 0.05, "live": False},
                {"key": "detect.iou", "label": "IoU", "value": cfg["detect_iou"],
                 "min": 0.3, "max": 0.9, "step": 0.05, "live": False}]},
            {"step": "3 · Body ReID", "desc": "Appearance embedding — applies to real footage", "fields": [
                {"key": "reid.sim_threshold", "label": "Match similarity", "value": cfg["reid_sim"],
                 "min": 0.3, "max": 0.95, "step": 0.05, "live": False}]},
            {"step": "7 · Cross-camera fusion", "desc": "Merge co-temporal sightings → one global ID", "fields": [
                {"key": "fusion.time_window", "label": "Time window (s)", "value": p.fusion_time_window,
                 "min": 1, "max": 30, "step": 1, "live": True},
                {"key": "fusion.sim_threshold", "label": "Similarity", "value": p.fusion_sim_threshold,
                 "min": 0.1, "max": 0.95, "step": 0.05, "live": True}]},
            {"step": "6 · Sessions (entry↔exit)", "desc": "Dwell matching + review band", "fields": [
                {"key": "sessions.match_threshold", "label": "Auto-match ≥", "value": p.match_threshold,
                 "min": 0.5, "max": 0.99, "step": 0.01, "live": True},
                {"key": "sessions.review_threshold", "label": "Review band ≥", "value": p.review_threshold,
                 "min": 0.3, "max": 0.9, "step": 0.01, "live": True}]},
            {"step": "Employee gallery", "desc": "Match vs enrolled staff (face/body)", "fields": [
                {"key": "employee.threshold", "label": "Employee match ≥", "value": p.employee_threshold,
                 "min": 0.3, "max": 0.95, "step": 0.01, "live": True}]},
            {"step": "8 · Grouping", "desc": "Arrival-window clustering", "fields": [
                {"key": "grouping.gap_sec", "label": "Group gap (s)", "value": p.group_gap_sec,
                 "min": 1, "max": 60, "step": 1, "live": True}]},
        ]

    def training_view(self) -> dict:
        counts = {"is_employee": 0, "demographics": 0, "group": 0}
        for c in self.store.all():
            if c.field in counts:
                counts[c.field] += 1
        registry = [{"version": mv.version, "score": round(mv.metrics.get("score", 0) * 100),
                     "active": mv.version == self.registry.active_version}
                    for mv in self.registry.versions]
        return {
            "settings": self.train_settings,
            "labels_total": len(self.store),
            "labels_since_train": len(self.store) - self._labels_at_last_train,
            "registry": registry,
            "last_result": self.last_result,
            "where": "Indian GPU cloud (batch) — same GPUs that process the nightly footage",
            "models": [
                {"name": "Detector (YOLO11x)", "method": "fine-tune on corrected boxes", "pending": 0},
                {"name": "Body ReID (CLIP-ReID)", "method": "metric learning on same/diff pairs", "pending": 0},
                {"name": "Age/Gender (MiVOLO)", "method": "classification head fine-tune", "pending": counts["demographics"]},
                {"name": "Employee gallery", "method": "enroll embeddings", "pending": counts["is_employee"]},
                {"name": "Grouping", "method": "tune gap / learned classifier", "pending": counts["group"]},
            ],
        }

    def snapshot(self) -> dict:
        report, gt, mgr = run_pipeline(params=self.params)
        score, checks = day_score(report, gt)
        for k, sheet in report["employees"]["timesheets"].items():
            sheet["name"] = self.employees.get(k, "")
        pending = [c for c in collect_corrections(mgr) if c.item_id not in self.applied_ids]
        self._pending = {c.item_id: c for c in pending}
        return {
            "report": report,
            "score": round(score * 100),
            "checks": checks,
            "cards": build_review_cards(mgr, pending),
            "active_version": self.registry.active_version,
            "n_versions": len(self.registry.versions),
            "applied_labels": len(self.store),
            "params": self.params_view(),
            "training": self.training_view(),
            "employees": dict(self.employees),
        }

    # ---- mutations ------------------------------------------------------
    def apply_correction(self, card_id: str, value: dict | None) -> bool:
        if card_id not in self._pending:
            self.snapshot()
        base = self._pending.get(card_id)
        if base is None:
            return False
        value = value or {}
        if base.field == "is_employee":
            if not value.get("is_employee", True):  # reviewer confirms customer -> no enrollment
                self.applied_ids.add(card_id)
                return True
            eid = value.get("employee_id") or base.employee_id
            name = (value.get("name") or "").strip()
            if name:
                self.employees[eid] = name  # record / update the staff directory
            corr = Correction(item_id=base.item_id, field="is_employee", before=False, after=True,
                              embedding=base.embedding, employee_id=eid, subject=base.subject)
        elif base.field == "demographics":
            after = {k: value[k] for k in ("gender", "age_bucket") if k in value} or dict(base.after or {})
            corr = Correction(item_id=base.item_id, field="demographics", before=base.before,
                              after=after, subject=base.subject)
        else:
            corr = base
        self.store.add_round([corr])
        self.applied_ids.add(card_id)
        return True

    def update_params(self, patch: dict) -> None:
        changes, cfg = {}, self.pipeline_config
        live = {"fusion.time_window": "fusion_time_window", "fusion.sim_threshold": "fusion_sim_threshold",
                "sessions.match_threshold": "match_threshold", "sessions.review_threshold": "review_threshold",
                "employee.threshold": "employee_threshold", "grouping.gap_sec": "group_gap_sec"}
        config = {"detect.confidence": "detect_conf", "detect.iou": "detect_iou", "reid.sim_threshold": "reid_sim"}
        for k, v in (patch or {}).items():
            if k in live:
                changes[live[k]] = float(v)
            elif k in config:
                cfg[config[k]] = float(v)
        if changes:
            self.params = replace(self.params, **changes)

    def update_train_settings(self, patch: dict) -> None:
        for k in ("cadence", "min_new_labels", "target", "auto_promote"):
            if k in (patch or {}):
                self.train_settings[k] = patch[k]

    def retrain(self) -> dict:
        _, gt, _ = run_pipeline(params=self.params)
        before_report, _, _ = run_pipeline(params=self.params)
        before, _ = day_score(before_report, gt)
        candidate = train_candidate(self.params, self.store)
        cand_report, _, _ = run_pipeline(params=candidate)
        after, after_checks = day_score(cand_report, gt)

        target = float(self.train_settings["target"])
        eligible = after >= target and after > before
        result = {"before": round(before * 100), "after": round(after * 100),
                  "eligible": eligible, "auto_promote": bool(self.train_settings["auto_promote"]),
                  "promoted": False, "candidate_version": None}
        if eligible:
            mv = self.registry.register(candidate, {"score": after, "checks": after_checks})
            result["candidate_version"] = mv.version
            if self.train_settings["auto_promote"]:
                self.registry.promote(mv.version)
                self.params = candidate
                result["promoted"] = True
        result["active_version"] = self.registry.active_version
        self.last_result = result
        self._labels_at_last_train = len(self.store)
        return result

    def promote(self, version: int) -> bool:
        mv = next((m for m in self.registry.versions if m.version == version), None)
        if mv is None:
            return False
        self.registry.promote(version)
        self.params = mv.params
        return True


app = FastAPI(title="Showroom Review Console")
state = DashboardState()

# Real-footage HITL console (visits from actual L1/L4 on real clips) lives alongside the
# original synthetic dashboard. Browse it at /review.
from dashboard.hitl_api import router as hitl_router  # noqa: E402

app.include_router(hitl_router)


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/review", response_class=HTMLResponse)
def review_page():
    return (STATIC / "review.html").read_text(encoding="utf-8")


@app.get("/api/state")
def api_state():
    return state.snapshot()


@app.post("/api/correct")
def api_correct(body: dict):
    ok = state.apply_correction(body.get("id", ""), body.get("value"))
    return {"ok": ok, "state": state.snapshot()}


@app.post("/api/params")
def api_params(body: dict):
    state.update_params(body.get("patch", body))
    return state.snapshot()


@app.post("/api/train_settings")
def api_train_settings(body: dict):
    state.update_train_settings(body)
    return state.snapshot()


@app.post("/api/retrain")
def api_retrain():
    result = state.retrain()
    return {"result": result, "state": state.snapshot()}


@app.post("/api/promote")
def api_promote(body: dict):
    ok = state.promote(int(body.get("version", 0)))
    return {"ok": ok, "state": state.snapshot()}


@app.get("/api/sources")
def api_sources():
    return state.sources_view()


@app.post("/api/sources")
def api_sources_save(body: dict):
    return state.save_sources(body)


@app.post("/api/sources/test")
def api_sources_test():
    return {"results": state.test_sources()}


@app.post("/api/reset")
def api_reset():
    state.reset()
    return state.snapshot()
