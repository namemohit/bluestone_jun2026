-- ===========================================================================
--  BlueStone Showroom — HITL people-tracker database
--  Lives in its OWN `showroom` schema inside the shared (yantrai) Supabase
--  project, so it is fully ISOLATED: nothing here references, reads, or writes
--  any public/yantrai object. Drop the whole thing with `drop schema showroom
--  cascade;` and yantrai is untouched.
--
--  Run once:  psql "$SUPABASE_DB_URL" -f supabase/schema.sql
--  Then expose it to the API:  Supabase ▸ Settings ▸ API ▸ Exposed schemas ▸ add "showroom"
-- ===========================================================================
create schema if not exists showroom;

-- ---- stores ---------------------------------------------------------------
create table if not exists showroom.stores (
  id             text primary key,                       -- 's14'
  name           text not null,
  timezone       text not null default 'Asia/Kolkata',
  clock_offset_s integer not null default 0,             -- OSD->IST seconds (20315 for the Jun-3 NVR)
  created_at     timestamptz not null default now()
);

-- ---- windows: one processed slice (an hour) — the unit of the HITL loop ---
create table if not exists showroom.windows (
  id            text primary key,                        -- 's14:2026-06-03_18'
  store_id      text not null references showroom.stores(id),
  date          date not null,
  start_ist     timestamptz not null,
  end_ist       timestamptz,                             -- null for an open/partial slice
  status        text not null default 'processing'       -- processing | ready | reviewed
                  check (status in ('processing','ready','reviewed')),
  params        jsonb not null default '{}'::jsonb,      -- min_sim, occ_floor, interior, fps, ...
  model_version integer,
  source        jsonb not null default '{}'::jsonb,      -- l1 dirs / clip paths it ran on
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now()
);

-- ---- events: EVERY entry/exit crossing, matched or not --------------------
--   role lets the reviewer act on the leftovers (open / pre_exit) and on staff.
create table if not exists showroom.events (
  id          bigint generated always as identity primary key,
  window_id   text not null references showroom.windows(id) on delete cascade,
  track       integer not null,                          -- entry-cam (C05) track id
  direction   text not null check (direction in ('in','out')),
  ts_ist      timestamptz not null,
  crop_url    text,                                       -- -> Storage bucket showroom-crops
  role        text not null default 'unmatched'          -- matched | open | pre_exit | staff
                check (role in ('matched','open','pre_exit','staff','unmatched')),
  created_at  timestamptz not null default now(),
  unique (window_id, track, direction)
);

-- ---- detections: L1 raw tracks (every camera) — regenerable snapshot for the detections view ----
--   + attendance (a staffer's in_track joins here for first/last sighting). Replaced per window push.
create table if not exists showroom.detections (
  id         bigint generated always as identity primary key,
  window_id  text not null references showroom.windows(id) on delete cascade,
  camera     text not null,                                 -- 'C05' (door) | 'C11' | 'C14' (interior)
  track      integer not null,
  first_ist  timestamptz,
  last_ist   timestamptz,
  dur_s      numeric,
  frames     integer,
  crop_url   text,
  created_at timestamptz not null default now()
);
create index if not exists detections_by_window on showroom.detections (window_id);

-- ---- visits: an IN<->OUT pairing = one customer visit ---------------------
create table if not exists showroom.visits (
  window_id    text not null references showroom.windows(id) on delete cascade,
  id           text not null,                            -- '203-329' (in_track-out_track)
  in_track     integer not null,
  out_track    integer not null,
  in_ist       timestamptz not null,
  out_ist      timestamptz not null,
  dwell_s      numeric not null,
  how          text not null,                            -- match method / sim, human-readable
  confidence   numeric,
  uncertainty  numeric,                                  -- active-learning rank key
  status       text not null default 'auto' check (status in ('auto','needs_review')),
  in_crop_url  text,
  out_crop_url text,
  is_customer  boolean not null default true,
  model_version integer,
  updated_at   timestamptz not null default now(),
  primary key (window_id, id)
);
create index if not exists visits_review_order on showroom.visits (window_id, uncertainty desc);

-- ---- labels: human verdicts — APPEND-ONLY audit trail ---------------------
--   never updated/deleted; the matcher compiles the latest verdict per visit.
create table if not exists showroom.labels (
  id         bigint generated always as identity primary key,
  window_id  text not null references showroom.windows(id) on delete cascade,
  visit_id   text not null,                              -- visit id OR 'open-<track>'
  verdict    text not null check (verdict in ('confirm','reject','employee','false_detection','reset')),
  reason     text default '',
  in_track   integer,
  out_track  integer,
  employee_id integer,                                      -- which staffer (set on 'employee' verdicts)
  reviewer   text not null default 'human',                 -- 'human' | 'auto' (gallery-recognised)
  created_at timestamptz not null default now()
);
create index if not exists labels_by_window on showroom.labels (window_id, created_at desc);

-- the "current truth" = newest verdict per (window, visit)
create or replace view showroom.latest_labels as
select distinct on (window_id, visit_id) *
from showroom.labels
order by window_id, visit_id, created_at desc;

-- ---- metrics: precision over time (the curve that should climb) -----------
create table if not exists showroom.metrics (
  id            bigint generated always as identity primary key,
  window_id     text not null references showroom.windows(id) on delete cascade,
  ts            timestamptz not null default now(),
  visits        integer, reviewed integer, confirmed integer, rejected integer,
  precision     numeric,
  model_version integer
);

-- ---- model_versions: registry for promote / rollback ----------------------
create table if not exists showroom.model_versions (
  version    integer generated always as identity primary key,
  kind       text not null default 'thresholds'          -- thresholds | osnet_finetune | gallery
               check (kind in ('thresholds','osnet_finetune','gallery')),
  params     jsonb not null default '{}'::jsonb,
  score      numeric,
  trained_on integer default 0,                          -- # human labels behind this version
  active     boolean not null default false,
  notes      text default '',
  created_at timestamptz not null default now()
);

-- ---- employees: enrolled staff gallery (auto-filtered from counts) --------
create table if not exists showroom.employees (
  id          integer generated always as identity primary key,
  store_id    text not null references showroom.stores(id),
  name        text,
  code        text,                                       -- 'S<id>' display code (Staff #N)
  crop_urls   text[] default '{}',
  embedding   jsonb,                                      -- swap to pgvector(512) for in-DB ReID later
  enrolled_at timestamptz not null default now()
);

-- ---- employee_gallery: many OSNet embeddings per staffer -> robust cross-hour auto-recognition --
create table if not exists showroom.employee_gallery (
  id            bigint generated always as identity primary key,
  employee_id   integer not null references showroom.employees(id) on delete cascade,
  store_id      text not null,
  embedding     jsonb not null,                           -- 512-d OSNet vector as a json list
  crop_url      text,
  source_window text,
  source_track  integer,
  added_at      timestamptz not null default now()
);
create index if not exists gallery_by_store on showroom.employee_gallery (store_id);

-- ---- keep windows.updated_at fresh on every change ------------------------
create or replace function showroom.touch_updated_at() returns trigger
  language plpgsql as $$ begin new.updated_at = now(); return new; end $$;
drop trigger if exists trg_windows_touch on showroom.windows;
create trigger trg_windows_touch before update on showroom.windows
  for each row execute function showroom.touch_updated_at();

-- ---- dashboard overview view ----------------------------------------------
create or replace view showroom.window_summary as
select w.id, w.store_id, w.date, w.start_ist, w.status, w.model_version,
       count(v.*)                              as visits,
       count(v.*) filter (where v.status='needs_review') as needs_review,
       (select count(distinct l.visit_id) from showroom.labels l where l.window_id = w.id) as reviewed,
       w.updated_at
from showroom.windows w
left join showroom.visits v on v.window_id = w.id
group by w.id;

-- ---- idempotent column adds: re-run this file to sync an already-created DB ----
alter table showroom.labels    add column if not exists employee_id integer;
alter table showroom.employees add column if not exists code text;
-- the labels verdict check predates 'reset'; widen it to match VERDICTS in hitl/store.py
alter table showroom.labels drop constraint if exists labels_verdict_check;
alter table showroom.labels add constraint labels_verdict_check
  check (verdict in ('confirm','reject','employee','false_detection','reset'));
