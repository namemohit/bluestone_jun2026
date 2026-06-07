"""NVR RTSP URL building + credential masking."""
import pytest

from ingest.nvr import build_rtsp_url, masked

NVR = {"brand": "hikvision", "host": "10.0.0.5", "rtsp_port": 554,
       "username": "admin", "password": "pass"}


def test_hikvision_main_channel():
    url = build_rtsp_url(NVR, {"channel": 1, "stream": "main"})
    assert url == "rtsp://admin:pass@10.0.0.5:554/Streaming/Channels/101"


def test_hikvision_sub_channel():
    url = build_rtsp_url(NVR, {"channel": 3, "stream": "sub"})
    assert url == "rtsp://admin:pass@10.0.0.5:554/Streaming/Channels/302"


def test_dahua_main_channel():
    nvr = {**NVR, "brand": "dahua"}
    url = build_rtsp_url(nvr, {"channel": 2, "stream": "main"})
    assert url == "rtsp://admin:pass@10.0.0.5:554/cam/realmonitor?channel=2&subtype=0"


def test_source_override_wins():
    url = build_rtsp_url(NVR, {"source": "rtsp://x/y", "channel": 1})
    assert url == "rtsp://x/y"


def test_generic_without_source_raises():
    with pytest.raises(ValueError):
        build_rtsp_url({"brand": "generic", "host": "h"}, {"channel": 1})


def test_password_is_url_encoded():
    nvr = {**NVR, "password": "p@s/s"}
    url = build_rtsp_url(nvr, {"channel": 1, "stream": "main"})
    assert "p%40s%2Fs" in url


def test_masked_hides_credentials():
    m = masked("rtsp://admin:secret@10.0.0.5:554/Streaming/Channels/101")
    assert "secret" not in m and "admin" not in m and m.endswith("/101")
