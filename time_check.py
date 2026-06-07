"""Verify the NVR's clock is in sync with real IST.

Queries ISAPI /System/time for the NVR's current local time and NTP config, and
prints the laptop's current time alongside. If they agree (and NTP is enabled),
the burned-in OSD timestamps on recordings are true IST.
Credentials are read from configs/nvr.json (never printed).
"""
import json
from datetime import datetime, timezone, timedelta

import requests
from requests.auth import HTTPDigestAuth

IST = timezone(timedelta(hours=5, minutes=30))

cfg = json.load(open("configs/nvr.json"))
ip = cfg["nvr_public_ip"]
dev = next(d for d in cfg["devices"] if d["name"] == "NVR 2")
base = f"http://{ip}:{dev['public_port']}"
auth = HTTPDigestAuth(dev["username"], dev["password"])

for path in ("/ISAPI/System/time", "/ISAPI/System/time/ntpServers"):
    try:
        r = requests.get(base + path, auth=auth, timeout=15)
        print(f"\n=== GET {path}  -> HTTP {r.status_code} ===")
        print(r.text.strip())
    except Exception as e:
        print(f"\n=== GET {path}  -> ERROR {e}")

print("\n=== reference clocks ===")
print("laptop now (local):", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("laptop now (IST)  :", datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S %Z"))
