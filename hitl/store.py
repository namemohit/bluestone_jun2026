"""HITL store: processed visits + human labels + the compiled matcher-feedback, behind ONE
interface so the dashboard and the loop never care where data lives.

`LocalStore` (JSON files under outputs/<window>/) runs the whole loop offline today. A
`SupabaseStore` with the same methods drops in later for the hosted Cloud Run dashboard --
same calls, different backend, zero UI changes.

A *window* is one processed slice (an hour), keyed like '2026-06-03_18'. The flow per window:
  l4_visits  ->  visits.json            (machine result, uncertainty-ranked)
  human      ->  add_label(...)         (tick / cross / "that's staff")
  store      ->  feedback.json          (labels compiled to cannot_link / must_link / employees)
  l4_visits --feedback  ->  visits.json (re-routed, FREE, no GPU)   <-- the learning loop
"""
from __future__ import annotations

import json
import pathlib

VERDICTS = ("confirm", "reject", "employee", "false_detection", "reset")  # reset = undo (back to unlabelled)


class LocalStore:
    def __init__(self, root: str = "outputs"):
        self.root = pathlib.Path(root)

    def _wdir(self, window: str) -> pathlib.Path:
        d = self.root / window
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---- reads ----------------------------------------------------------
    def get_visits(self, window: str) -> dict:
        p = self._wdir(window) / "visits.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"visits": [], "counts": {}}

    def get_labels(self, window: str) -> list[dict]:
        p = self._wdir(window) / "labels.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []

    # ---- writes ---------------------------------------------------------
    def add_label(self, window: str, visit_id: str, verdict: str, *, reason: str = "",
                  in_track=None, out_track=None, reviewer: str = "human", employee_id=None) -> dict:
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
        labels = [l for l in self.get_labels(window) if l["visit_id"] != visit_id]  # last wins
        rec = {"visit_id": visit_id, "verdict": verdict, "reason": reason,
               "in_track": in_track, "out_track": out_track, "reviewer": reviewer,
               "employee_id": employee_id}
        labels.append(rec)
        (self._wdir(window) / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
        return rec

    # ---- compile labels -> matcher feedback -----------------------------
    def feedback(self, window: str) -> dict:
        cannot, must, employees = [], [], []
        for l in self.get_labels(window):
            it, ot = l.get("in_track"), l.get("out_track")
            v = l["verdict"]
            if v == "reject" and it is not None and ot is not None:
                cannot.append([it, ot])          # this IN and OUT are NOT the same person
            elif v == "confirm" and it is not None and ot is not None:
                must.append([it, ot])            # lock this pairing
            elif v == "employee" and it is not None:
                employees.append(it)             # drop staff from customer counts
        return {"cannot_link": cannot, "must_link": must, "employees": employees}

    def write_feedback(self, window: str) -> str:
        p = self._wdir(window) / "feedback.json"
        p.write_text(json.dumps(self.feedback(window), indent=2), encoding="utf-8")
        return str(p)

    # ---- progress metric (the curve that should climb hour over hour) ---
    def metrics(self, window: str) -> dict:
        visits = self.get_visits(window).get("visits", [])
        labels = {l["visit_id"]: l["verdict"] for l in self.get_labels(window) if l["verdict"] != "reset"}
        reviewed = [v for v in visits if v["id"] in labels]
        confirmed = sum(1 for v in reviewed if labels[v["id"]] in ("confirm", "employee"))
        return {"visits": len(visits), "reviewed": len(reviewed),
                "confirmed": confirmed, "rejected": len(reviewed) - confirmed,
                "precision": round(confirmed / len(reviewed), 3) if reviewed else None,
                "unreviewed": len(visits) - len(reviewed)}

    def list_windows(self) -> list[dict]:
        out = []
        for d in sorted(self.root.glob("*/")):
            if (d / "visits.json").exists():
                out.append({"window": d.name, **self.metrics(d.name)})
        return out

    def get_detections(self, window: str) -> list[dict]:
        """L1 raw detections (all cameras) read from the window's local L1 dirs."""
        wj = self.root / window / "window.json"
        if not wj.exists():
            return []
        cfg = json.loads(wj.read_text(encoding="utf-8"))
        dirs = [(cfg.get("l1"), "C05")] + [(d, pathlib.Path(d).name.replace("L1_", "")) for d in cfg.get("interior", [])]
        out = []
        for d, cam in dirs:
            tj = pathlib.Path(d) / "tracks.json" if d else None
            if not tj or not tj.exists():
                continue
            for t in json.loads(tj.read_text(encoding="utf-8")).get("tracks", []):
                out.append({"camera": cam, "track": t["track"], "ist": t["first_ist"],
                            "dur_s": round(t.get("last_ts", 0) - t.get("first_ts", 0), 1),
                            "frames": t.get("frames"), "crop": t.get("crop", "").replace("\\", "/")})
        return sorted(out, key=lambda x: x["ist"])

    # --- interchangeability with SupabaseStore: local IS the source of truth, so these are no-ops
    def record_metrics(self, window: str) -> dict:
        return self.metrics(window)

    def sync(self, window: str, store_id: str = "s14") -> None:
        return None

    def upload_crops(self, window: str) -> int:
        return 0  # local dashboard serves crops from disk

    # --- employee roster: needs the DB; stubs so a pure-local deploy doesn't crash ---
    def list_employees(self, store_id: str = "s14") -> list[dict]:
        return []

    def create_employee(self, store_id: str = "s14", name=None) -> dict:
        return {"id": 0, "code": "S0", "name": name}

    def rename_employee(self, emp_id: int, name: str) -> None:
        return None

    def enroll_staff(self, employee_id, store_id, embedding, crop_url=None, window=None, track=None) -> None:
        return None

    def get_gallery(self, store_id: str = "s14") -> list[dict]:
        return []

    def attendance(self, store_id: str = "s14", date=None) -> list[dict]:
        return []
