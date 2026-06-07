"""ISAPI footage ingestion: channel/track math, IST->UTC, search XML, parse, mocked download."""
from ingest.nvr_isapi import (
    IsapiFootageClient,
    NvrDevice,
    build_search_xml,
    channel_from_camera,
    load_devices,
    parse_segments,
    to_utc_z,
    track_id,
)

SAMPLE_SEARCH = """<?xml version="1.0" encoding="UTF-8"?>
<CMSearchResult xmlns="http://www.hikvision.com/ver20/XMLSchema">
  <matchList>
    <searchMatchItem>
      <timeSpan><startTime>2026-06-03T04:30:00Z</startTime><endTime>2026-06-03T05:00:00Z</endTime></timeSpan>
      <mediaSegmentDescriptor><playbackURI>rtsp://h/Streaming/tracks/501?starttime=20260603T043000Z</playbackURI></mediaSegmentDescriptor>
    </searchMatchItem>
    <searchMatchItem>
      <timeSpan><startTime>2026-06-03T05:00:00Z</startTime><endTime>2026-06-03T05:30:00Z</endTime></timeSpan>
      <mediaSegmentDescriptor><playbackURI>rtsp://h/seg2</playbackURI></mediaSegmentDescriptor>
    </searchMatchItem>
  </matchList>
</CMSearchResult>"""


class FakeResp:
    def __init__(self, status=200, text="", chunks=None, ct="video/mp4"):
        self.status_code = status
        self.text = text
        self._chunks = chunks or []
        self.headers = {"Content-Type": ct}

    def iter_content(self, chunk_size=1):
        for c in self._chunks:
            yield c


class FakeSession:
    def __init__(self, post=None, get=None):
        self._post = list(post or [])
        self._get = get
        self.calls = []

    def post(self, url, **kw):
        self.calls.append(("POST", url))
        return self._post.pop(0)

    def get(self, url, **kw):
        self.calls.append(("GET", url))
        return self._get


def test_channel_and_track():
    assert channel_from_camera("C05") == 5
    assert channel_from_camera("11") == 11
    assert (track_id(5), track_id(11), track_id(14)) == (501, 1101, 1401)


def test_ist_to_utc_z():
    assert to_utc_z("2026-06-03T10:00:00") == "2026-06-03T04:30:00Z"
    assert to_utc_z("2026-06-03T22:00:00") == "2026-06-03T16:30:00Z"


def test_search_xml():
    xml = build_search_xml(5, "2026-06-03T04:30:00Z", "2026-06-03T16:30:00Z")
    assert "<trackID>501</trackID>" in xml
    assert "<startTime>2026-06-03T04:30:00Z</startTime>" in xml
    assert "<endTime>2026-06-03T16:30:00Z</endTime>" in xml


def test_parse_segments():
    segs = parse_segments(SAMPLE_SEARCH)
    assert len(segs) == 2
    assert segs[0]["start_time"] == "2026-06-03T04:30:00Z"
    assert segs[0]["playback_uri"].startswith("rtsp://h/Streaming/tracks/501")


def test_load_devices_public():
    cfg = {"nvr_public_ip": "202.0.0.1", "use_public_ip": True, "devices": [
        {"name": "NVR 2", "internal_ip": "172.16.204.100", "internal_port": 88,
         "public_port": 88, "username": "u", "password": "p", "channels": [5, 11, 14]}]}
    devs = load_devices(cfg)
    assert "NVR02" in devs
    assert devs["NVR02"].host == "202.0.0.1" and devs["NVR02"].port == 88


def _dev():
    return NvrDevice("NVR02", "NVR 2", "202.0.0.1", 88, "u", "p", [5, 11, 14])


def test_search_segments_mocked():
    sess = FakeSession(post=[FakeResp(200, SAMPLE_SEARCH)])
    segs = IsapiFootageClient(_dev(), session=sess).search_segments(5, "2026-06-03T04:30:00Z", "2026-06-03T16:30:00Z")
    assert len(segs) == 2 and segs[0]["segment_number"] == 1
    assert sess.calls[0][1].endswith("/ISAPI/ContentMgmt/search")


def test_download_segment_writes_file(tmp_path):
    sess = FakeSession(post=[FakeResp(200, chunks=[b"abc", b"def"], ct="video/mp4")])
    out = tmp_path / "seg.mp4"
    size = IsapiFootageClient(_dev(), session=sess).download_segment("rtsp://h/seg", out)
    assert size == 6 and out.read_bytes() == b"abcdef"


def test_download_falls_back_to_get(tmp_path):
    # POST download returns 500 -> direct GET of the playback URI succeeds
    sess = FakeSession(post=[FakeResp(500, text="err")], get=FakeResp(200, chunks=[b"xyz"], ct="video/mp4"))
    out = tmp_path / "seg.mp4"
    size = IsapiFootageClient(_dev(), session=sess).download_segment("http://h/seg.mp4", out)
    assert size == 3 and out.read_bytes() == b"xyz"
