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

-- ---- annotations: HUMAN ground-truth per detection (close-the-day) + durable training labels ----
--   one human allocation per (window,camera,track); latest wins. Doubles as the ReID/YOLO dataset.
create table if not exists showroom.annotations (
  id           bigint generated always as identity primary key,
  window_id    text not null references showroom.windows(id) on delete cascade,
  camera       text not null,                                -- 'C05' | 'C11' | 'C14'
  track        integer not null,
  crop_url     text,
  category     text not null
                 check (category in ('customer','staff','not_person','passby','duplicate')),
  employee_id  integer,                                      -- which staffer (category='staff')
  duplicate_of integer,                                      -- another track (category='duplicate')
  embedding    jsonb,                                        -- 512-d OSNet vector (from the cache)
  reviewer     text not null default 'human',
  created_at   timestamptz not null default now()
);
create index if not exists annotations_by_window on showroom.annotations (window_id);
-- the current human truth = newest allocation per (window,camera,track)
create or replace view showroom.latest_annotations as
select distinct on (window_id, camera, track) *
from showroom.annotations
order by window_id, camera, track, created_at desc;

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

-- ---- published_reports: the FINALIZED client-facing snapshot per period --------------------
--   "Publish" freezes the day's (or hour's) report so the read-only public dashboard surfaces it
--   to the B2B customer. Append-only; latest per (period,scope) wins. No redeploy needed.
create table if not exists showroom.published_reports (
  id            bigint generated always as identity primary key,
  store_id      text not null,
  period        text not null,                                -- 'YYYY-MM-DD' (day) or 'YYYY-MM-DD_HHMM' (hour)
  scope         text not null default 'day' check (scope in ('day','hour')),
  report        jsonb not null,                               -- frozen customers/groups/employees snapshot
  model_version integer,                                      -- model_versions row behind these numbers
  published_by  text not null default 'human',
  published_at  timestamptz not null default now()
);
create index if not exists published_by_period on showroom.published_reports (store_id, period, scope);
-- the live client view = newest snapshot per (period,scope)
create or replace view showroom.latest_published as
select distinct on (store_id, period, scope) *
from showroom.published_reports
order by store_id, period, scope, published_at desc;

-- ---- idempotent column adds: re-run this file to sync an already-created DB ----
alter table showroom.labels    add column if not exists employee_id integer;
alter table showroom.employees add column if not exists code text;
-- the labels verdict check predates 'reset'; widen it to match VERDICTS in hitl/store.py
alter table showroom.labels drop constraint if exists labels_verdict_check;
alter table showroom.labels add constraint labels_verdict_check
  check (verdict in ('confirm','reject','employee','false_detection','reset'));

-- ---- Phase 2: permanent staff number (immutable; S1 always means ONE person) -----------------
--   staff_no = 1..n by enrollment order; a future hire takes max(staff_no)+1. Unlike the old
--   rank-by-id display, deleting an employee never renumbers the others.
alter table showroom.employees add column if not exists staff_no integer;
with ranked as (select id, row_number() over (order by id) rn
                from showroom.employees where store_id = 's14')
update showroom.employees e set staff_no = r.rn
  from ranked r where e.id = r.id and e.staff_no is null;
create unique index if not exists employees_staff_no_uniq on showroom.employees (store_id, staff_no);

-- ---- Phase 2: person_contexts — durable per-day PID registry (the enrichable "context") --------
--   Frozen at /publish so #C / #G stop drifting between views after the day is finalized. Also the
--   home each PID's accumulated evidence will hang off (exit decision, demographics, ReID emb later).
create table if not exists showroom.person_contexts (
  id          bigint generated always as identity primary key,
  store_id    text not null default 's14',
  date        date not null,
  kind        text not null default 'customer' check (kind in ('customer','staff')),
  pid_no      integer not null,                              -- frozen #C (customer) within the day
  group_no    integer,                                       -- frozen #G
  employee_id integer,                                       -- set when kind='staff'
  window_id   text,                                          -- canonical (earliest) entry window
  track       integer,                                       -- canonical entry door track
  in_ist      time,
  out_ist     time,
  dwell_s     numeric,
  exit_src    text,                                          -- matched | presumed | null
  frozen_at   timestamptz not null default now(),
  meta        jsonb not null default '{}'::jsonb
);
create index if not exists person_contexts_by_day on showroom.person_contexts (store_id, date);
create unique index if not exists person_contexts_pid_uniq
  on showroom.person_contexts (store_id, date, kind, pid_no);
