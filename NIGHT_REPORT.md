# 🌙 Overnight run — morning report

_Good morning. Here's everything I did while you slept, and what's waiting on you._

## TL;DR
- ✅ **All code pushed to GitHub** (`github.com/namemohit/bluestone_jun2026`, branch `main`) — secrets verified out.
- ✅ **Employee roster + auto-recognition + attendance** — finished, tested, working.
- ✅ **All 3 cameras prepped** (C05 / C11 / C14 day clips ready at 09:00 IST).
- 🟡 **Processing is running but slow** (~1.5 h per hour-of-footage on the 3050). You'll wake to the
  **early-afternoon hours** done, not the whole day — the full day can't finish in one night locally.
  See *Finishing the day* below.

## What's processed (check `/review`)
The resumable driver (`batch/run_overnight.py`) is tiling the open day into 60-min windows from the
store opening (11:22), **C05 + C11**, 3 fps, pushing each to Supabase as it finishes. It auto-skips
done windows, so it's safe to stop/restart.

- `11:22` ✅ (2 visits, 6 still-inside, 3 pre-window exits)
- `12:22` → `21:22` processing in order; expect ~4–6 hours done by morning.
- The empty shuttered morning (09–11:24) was **skipped on purpose** (0 customers).

## Features built tonight (all committed + tested)
1. **Mark identified staff** — on any person click **👤 Staff** → pick an existing staffer or "New
   staff" (Staff #1, #2…; add real names anytime with ✎). Fixes a real bug where enrollment was
   silently failing (numpy float32 wasn't JSON-serializable + a cache-key separator mismatch).
2. **Auto-recognition** — once you mark someone, their OSNet embedding is enrolled. Every hour
   processed/re-run afterward **auto-tags that same person** (🤖 badge) with no re-clicking, and
   drops them from the customer count. (`stack/l4_visits.py --gallery`, threshold `--staff-sim 0.6`,
   tunable if a customer is mis-tagged or a staffer is missed.)
3. **Attendance** — new section per staffer: **time in/out, sightings, # windows** across the day.
   I also fixed the pipeline to push L1 detections to the DB so this works for every hour.
4. **C14 ready** — third camera trimmed to `C14_day.ts` (verified to start exactly 09:00 IST). Fold
   it in for demographics / a second interior bridge by processing with `--interior C11 C14`.

## Finishing the day (your call)
The whole day × 3 cameras is **~30 h locally** — impossible overnight. Options when you're up:
- **Keep going locally** — the driver continues hour by hour; full day lands across the next ~day.
- **Cloud GPU pass (~$5–10, ~3 h)** for the entire day at once. **I did NOT spend any cloud money**
  while you slept (you'd said no cloud GPU). Say the word and I'll set it up.
- **Add C14** to remaining hours (richer, but ~50% slower per hour).

## Decisions I made for you (you were asleep)
- **No cloud spend** without a yes → local partial + cloud ready on request.
- **C05 + C11** for the overnight (covers more *hours* of customer counting; C14 prepped to fold in).
- **Skipped 09–11:24** (shuttered, empty).

## Known gaps / follow-ups
- Auto-staff currently shows on the **local** dashboard (reads the local L4 result); the public cloud
  dashboard shows customer counts but not the auto-staff overlay yet.
- `--staff-sim 0.6` is a first guess for cross-hour matching — we'll tune it on your real marks.
- Processing is the bottleneck, not accuracy — the 3050 runs ~real-time per camera.

## Dashboards
- **Local (you review here):** http://127.0.0.1:8000/review
- **Public (friends, faces hidden):** the Cloud Run URL from before (read-only).
