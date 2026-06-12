"""YantrAI store SSO — the *remote agent* side of REMOTE_AGENT_SPEC v1, implemented directly (no yantrai_sdk).

When the Vision Agent is opened from the YantrAI agentic store it is rendered in an <iframe>, and on the first
load the platform appends a short-lived signed token:  GET {remote_url}/?token=<token>

    token = b64url(payload) + "." + b64url(HMAC_SHA256(signing_key, b64url(payload)))
    payload claims = {u: username, c: workspace, exp: unix_secs (+300s), kid: client_id}

We verify it with our shared `signing_key`, mint our own longer-lived session cookie (`va_sess`, same envelope),
and make every response iframe-embeddable. The platform stores the SAME signing_key, so verification is offline.

ACTIVATION IS OPT-IN: the gate runs ONLY when `YANTRAI_SSO=1` AND `YANTRAI_SIGNING_KEY` is set. With neither, this
is a complete no-op — the local console (`dashboard/app.py` on :8050) is unchanged and never asks for a token.

Env:
    YANTRAI_SSO=1                 turn the gate ON (deployed/embedded only)
    YANTRAI_SIGNING_KEY=sk_...    HMAC secret (from backfill_publish_l2_agents.py output)
    YANTRAI_CLIENT_ID=cid_...     our app id (the token `kid`; used later for usage/billing)
    YANTRAI_PLATFORM_URL=...      default https://workspace.yantrailabs.com (for the no-secret verify fallback)
    VA_SESSION_TTL=43200          session-cookie lifetime, seconds (default 12h)
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time

FRAME_ANCESTOR = os.getenv("YANTRAI_PLATFORM_URL", "https://workspace.yantrailabs.com").rstrip("/")
_CSP = f"frame-ancestors 'self' {FRAME_ANCESTOR}"
_COOKIE = "va_sess"
_EXEMPT = ("/healthz",)


def _signing_key() -> str:
    return os.getenv("YANTRAI_SIGNING_KEY", "")


def sso_enabled() -> bool:
    """Gate is live ONLY when explicitly switched on AND a key is present — local dev stays open."""
    return os.getenv("YANTRAI_SSO", "") == "1" and bool(_signing_key())


def _session_ttl() -> int:
    try:
        return int(os.getenv("VA_SESSION_TTL", "43200"))
    except ValueError:
        return 43200


def _b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64u_dec(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _sign(body_b64: str, key: str) -> str:
    return _b64u(hmac.new(key.encode(), body_b64.encode(), hashlib.sha256).digest())


def mint_token(claims: dict, key: str | None = None) -> str:
    """Sign claims into the REMOTE_AGENT_SPEC envelope (used for our own `va_sess` cookie + tests)."""
    key = key or _signing_key()
    body = _b64u(json.dumps(claims, separators=(",", ":")).encode())
    return f"{body}.{_sign(body, key)}"


def verify_token(token: str | None, key: str | None = None) -> dict | None:
    """Return the claims dict {u,c,exp,kid} if the HMAC checks out and exp is in the future, else None."""
    key = key or _signing_key()
    if not token or not key or "." not in token:
        return None
    body, _, sig = token.partition(".")
    if not hmac.compare_digest(sig, _sign(body, key)):
        return None
    try:
        p = json.loads(_b64u_dec(body))
    except Exception:
        return None
    if int(p.get("exp", 0)) < int(time.time()):
        return None
    return p


def verify_sso_remote(token: str, platform_url: str | None = None) -> dict | None:
    """No-shared-secret fallback: ask the platform to verify (REMOTE_AGENT_SPEC §2)."""
    import urllib.parse
    import urllib.request
    base = (platform_url or FRAME_ANCESTOR).rstrip("/")
    url = f"{base}/api/agents/verify-sso?token=" + urllib.parse.quote(token or "")
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            d = json.loads(r.read().decode())
        return {"u": d.get("username"), "c": d.get("company_name")} if d.get("ok") else None
    except Exception:
        return None


def install_sso(app) -> bool:
    """Attach the SSO gate + iframe headers to a FastAPI app. No-op (returns False) unless sso_enabled().

    Behaviour when on:
      * `?token=` on any request  -> verify -> (re)mint a `va_sess` cookie (SameSite=None; Secure; HttpOnly)
      * else a valid `va_sess`     -> allow
      * else (and not exempt)      -> 401
      * every response             -> Content-Security-Policy: frame-ancestors ... ; drop X-Frame-Options
    """
    if not sso_enabled():
        return False
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    key, ttl = _signing_key(), _session_ttl()

    class _SSOGate(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            path = request.url.path
            fresh = None
            tok = request.query_params.get("token")
            user = verify_token(tok, key) if tok else None
            if user:
                fresh = mint_token({"u": user.get("u"), "c": user.get("c"),
                                    "kid": user.get("kid"), "exp": int(time.time()) + ttl}, key)
            else:
                user = verify_token(request.cookies.get(_COOKIE), key)
            if not user and not any(path.startswith(p) for p in _EXEMPT):
                return JSONResponse({"detail": "YantrAI SSO required"}, status_code=401,
                                    headers={"Content-Security-Policy": _CSP})
            resp = await call_next(request)
            resp.headers["Content-Security-Policy"] = _CSP
            if "x-frame-options" in resp.headers:           # never block the store iframe
                del resp.headers["x-frame-options"]
            if fresh:
                resp.set_cookie(_COOKIE, fresh, max_age=ttl, httponly=True, secure=True, samesite="none")
            return resp

    app.add_middleware(_SSOGate)
    return True
