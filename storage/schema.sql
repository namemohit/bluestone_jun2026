-- Showroom tracker schema (PostgreSQL + pgvector).
-- Phase 1 actively uses: cameras, crossing_events, tracks, embeddings.
-- The remaining tables are forward-looking stubs for Phases 2-3 (visitors, employees,
-- demographics) so the data model is stable as those features land.

CREATE EXTENSION IF NOT EXISTS vector;

-- Cameras / channels --------------------------------------------------------
CREATE TABLE IF NOT EXISTS cameras (
    id          TEXT PRIMARY KEY,
    role        TEXT NOT NULL DEFAULT 'common',   -- entry | exit | common
    source      TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Per-camera tracklets (one row per local track id) -------------------------
CREATE TABLE IF NOT EXISTS tracks (
    id          BIGSERIAL PRIMARY KEY,
    camera_id   TEXT REFERENCES cameras(id),
    local_id    BIGINT NOT NULL,           -- tracker id within this camera
    global_id   BIGINT,                    -- assigned by cross-camera fusion (Phase 1)
    first_ts    TIMESTAMPTZ,
    last_ts     TIMESTAMPTZ,
    is_employee BOOLEAN
);
CREATE INDEX IF NOT EXISTS idx_tracks_camera ON tracks(camera_id);
CREATE INDEX IF NOT EXISTS idx_tracks_global ON tracks(global_id);

-- Line-crossing events (the counting signal) --------------------------------
CREATE TABLE IF NOT EXISTS crossing_events (
    id          BIGSERIAL PRIMARY KEY,
    camera_id   TEXT NOT NULL,
    line_id     TEXT NOT NULL,
    track_id    BIGINT NOT NULL,
    ts          TIMESTAMPTZ NOT NULL,
    direction   TEXT NOT NULL,             -- in | out
    px          REAL,
    py          REAL,
    frame_idx   INTEGER,
    confidence  REAL DEFAULT 1.0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_cross_cam_ts ON crossing_events(camera_id, ts);
CREATE INDEX IF NOT EXISTS idx_cross_ts ON crossing_events(ts);

-- Appearance embeddings (tracklet-level), time-indexed for windowed matching --
-- vector dim 512 (OSNet). Change if your ReID model differs.
CREATE TABLE IF NOT EXISTS embeddings (
    id          BIGSERIAL PRIMARY KEY,
    camera_id   TEXT NOT NULL,
    track_id    BIGINT NOT NULL,
    ts          TIMESTAMPTZ NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'body',   -- body | face
    vec         VECTOR(512) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_emb_ts ON embeddings(ts);
-- ANN index for cosine similarity within a time window (see plan's gating design).
CREATE INDEX IF NOT EXISTS idx_emb_vec ON embeddings USING hnsw (vec vector_cosine_ops);

-- Visit sessions (one per global id; built in Phase 1 fusion) ----------------
CREATE TABLE IF NOT EXISTS visit_sessions (
    id          BIGSERIAL PRIMARY KEY,
    global_id   BIGINT,
    entered_at  TIMESTAMPTZ,
    exited_at   TIMESTAMPTZ,
    group_id    BIGINT,                    -- Phase 3 grouping
    is_employee BOOLEAN DEFAULT FALSE
);

-- Employees (enrolled) + in/out events --------------------------------------
CREATE TABLE IF NOT EXISTS employees (
    id          BIGSERIAL PRIMARY KEY,
    name        TEXT,
    active      BOOLEAN DEFAULT TRUE,
    enrolled_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS employee_events (
    id          BIGSERIAL PRIMARY KEY,
    employee_id BIGINT REFERENCES employees(id),
    ts          TIMESTAMPTZ NOT NULL,
    direction   TEXT NOT NULL,             -- in | out
    confidence  REAL
);
CREATE INDEX IF NOT EXISTS idx_emp_events ON employee_events(employee_id, ts);

-- Per-visitor demographics (Phase 2) ----------------------------------------
CREATE TABLE IF NOT EXISTS demographics (
    id          BIGSERIAL PRIMARY KEY,
    global_id   BIGINT,
    gender      TEXT,                      -- male | female | unknown
    gender_conf REAL,
    age_bucket  TEXT,                      -- e.g. 0-12 | 13-19 | 20-34 | 35-54 | 55+
    age_conf    REAL,
    ts          TIMESTAMPTZ DEFAULT now()
);
