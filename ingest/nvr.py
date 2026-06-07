"""Build per-channel RTSP URLs from an NVR config block.

Supports common brands; for anything else set brand=generic and provide an explicit
`source` rtsp:// URL per camera. Passwords are URL-encoded.
"""
from __future__ import annotations

from urllib.parse import quote


def build_rtsp_url(nvr: dict, camera: dict) -> str:
    """Return the RTSP URL for a camera, honoring an explicit `source` override."""
    override = camera.get("source")
    if override:
        return str(override)

    brand = (nvr.get("brand") or "generic").lower()
    host = nvr.get("host")
    port = int(nvr.get("rtsp_port", 554))
    user = nvr.get("username", "") or ""
    pwd = nvr.get("password", "") or ""
    channel = int(camera.get("channel", 1))
    stream = (camera.get("stream") or "main").lower()

    if not host:
        raise ValueError("nvr.host is required to build an RTSP URL")

    # Fully encode credentials (safe="") so '/', '@', ':' etc. in a password don't
    # break the RTSP URL's userinfo component.
    auth = f"{quote(user, safe='')}:{quote(pwd, safe='')}@" if user else ""

    if brand == "hikvision":
        # channel*100 + (01 main | 02 sub), e.g. ch1 main -> 101, ch3 sub -> 302
        sid = 1 if stream == "main" else 2
        return f"rtsp://{auth}{host}:{port}/Streaming/Channels/{channel * 100 + sid}"

    if brand == "dahua":
        subtype = 0 if stream == "main" else 1
        return (
            f"rtsp://{auth}{host}:{port}/cam/realmonitor"
            f"?channel={channel}&subtype={subtype}"
        )

    raise ValueError(
        f"brand={brand!r} has no URL template; set brand to 'hikvision'/'dahua' "
        f"or provide an explicit `source` rtsp:// URL on camera {camera.get('id')!r}."
    )


def masked(url: str) -> str:
    """Mask credentials in an RTSP URL for safe logging."""
    if "@" not in url or "//" not in url:
        return url
    scheme, rest = url.split("//", 1)
    creds, tail = rest.split("@", 1)
    return f"{scheme}//****:****@{tail}"
