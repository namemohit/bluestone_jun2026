"""Optional Postgres event writer.

Only imported when storage.enabled is true in the config, so the core pipeline runs
without psycopg installed. Requires `pip install "psycopg[binary]"`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from logic.events import CrossingEvent

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class EventWriter:
    def __init__(self, dsn: str, ensure_schema: bool = False):
        import psycopg  # lazy; optional dependency

        self.conn = psycopg.connect(dsn, autocommit=True)
        if ensure_schema:
            self.apply_schema()

    def apply_schema(self) -> None:
        sql = SCHEMA_PATH.read_text(encoding="utf-8")
        with self.conn.cursor() as cur:
            cur.execute(sql)

    def ensure_camera(self, camera: dict) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO cameras (id, role, source) VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET role = EXCLUDED.role
                """,
                (camera.get("id"), camera.get("role", "common"), str(camera.get("source") or "")),
            )

    def write_event(self, e: CrossingEvent) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO crossing_events
                    (camera_id, line_id, track_id, ts, direction, px, py, frame_idx, confidence)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    e.camera_id,
                    e.line_id,
                    int(e.track_id),
                    datetime.fromtimestamp(e.ts, tz=timezone.utc),
                    e.direction,
                    float(e.point[0]),
                    float(e.point[1]),
                    int(e.frame_idx),
                    float(e.confidence),
                ),
            )

    def write_session(self, s) -> None:
        """Persist a VisitSession (entered/exited/is_employee); dwell derived in queries."""
        def _dt(ts):
            return None if ts is None else datetime.fromtimestamp(ts, tz=timezone.utc)

        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO visit_sessions (entered_at, exited_at, is_employee) "
                "VALUES (%s, %s, %s)",
                (_dt(s.entry_ts), _dt(s.exit_ts), bool(s.is_employee)),
            )

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
