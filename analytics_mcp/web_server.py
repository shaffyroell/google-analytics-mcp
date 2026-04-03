# Copyright 2025 Google LLC All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HTTP-based MCP server with OAuth 2.0 authorization for cloud deployment.

Implements the MCP Authorization spec so Claude.ai can authenticate
automatically. Uses the Streamable HTTP transport (mcp>=1.6) which is
what modern MCP clients including Claude.ai expect.

Setup
-----
  GOOGLE_CLIENT_ID      – OAuth 2.0 Web Application client ID
  GOOGLE_CLIENT_SECRET  – OAuth 2.0 client secret
  SECRET_KEY            – Cookie signing secret (auto-generated if unset)
  BASE_URL              – Public URL, e.g. https://my-app.up.railway.app
  PORT                  – Listen port (default: 8080)

Add  <BASE_URL>/auth/callback  as an authorised redirect URI in Google
Cloud Console.
"""

import base64
import contextlib
import hashlib
import html
import logging
import os
import secrets
import time
from typing import Any, Dict

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.middleware.sessions import SessionMiddleware

import analytics_mcp.coordinator as coordinator
from analytics_mcp.tools.utils import _credentials_ctx

os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

logger = logging.getLogger(__name__)

_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/analytics.readonly",
]

# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------
_token_store: Dict[str, Dict[str, Any]] = {}   # bearer_token -> {credentials, email}
_auth_codes: Dict[str, Dict[str, Any]] = {}    # auth_code   -> {credentials, …, expires_at}
_pending_auths: Dict[str, Dict[str, Any]] = {} # oauth_session_id -> {client params, …}

# Scope fingerprint — stable 12-char hex used as OAuth realm and ETag on
# well-known responses, matching the pattern from google_workspace_mcp.
_SCOPE_FINGERPRINT: str = hashlib.sha256(
    " ".join(sorted(_SCOPES)).encode()
).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Environment variable '{name}' is not set.")
    return v


def _base_url(request: Request) -> str:
    explicit = os.environ.get("BASE_URL", "").rstrip("/")
    if explicit:
        return explicit
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.url.netloc)
    return f"{scheme}://{host}"


def _make_flow(redirect_uri: str) -> Flow:
    client_config = {
        "web": {
            "client_id": _require_env("GOOGLE_CLIENT_ID"),
            "client_secret": _require_env("GOOGLE_CLIENT_SECRET"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [redirect_uri],
        }
    }
    return Flow.from_client_config(client_config, scopes=_SCOPES, redirect_uri=redirect_uri)


def _creds_from_store(data: dict) -> Credentials:
    return Credentials(
        token=data["token"],
        refresh_token=data.get("refresh_token"),
        token_uri=data.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=data["client_id"],
        client_secret=data["client_secret"],
        scopes=data.get("scopes"),
    )


def _verify_pkce(challenge: str, verifier: str) -> bool:
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode() == challenge


_PAGE = """\
<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Google Analytics MCP</title>
<style>body{{font-family:system-ui,sans-serif;max-width:700px;margin:60px auto;padding:0 20px;color:#333}}
h1{{font-size:1.6rem}}pre{{background:#f5f5f5;padding:14px;border-radius:6px;overflow-x:auto;font-size:.9rem}}
a.btn{{display:inline-block;padding:10px 22px;background:#4285f4;color:#fff;border-radius:5px;text-decoration:none;font-weight:500}}
a.btn:hover{{background:#2b6fd4}}.label{{font-weight:600;margin-top:1.2rem;display:block}}</style>
</head><body>{body}</body></html>"""


def _page(body: str) -> HTMLResponse:
    return HTMLResponse(_PAGE.format(body=body))


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)

    # Session manager runs for the lifetime of the app.
    session_manager = StreamableHTTPSessionManager(
        app=coordinator.app,
        json_response=False,
        stateless=False,
        session_idle_timeout=1800,
    )

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        async with session_manager.run():
            yield

    app = FastAPI(title="Google Analytics MCP Server", lifespan=lifespan)
    app.add_middleware(SessionMiddleware, secret_key=secret_key, max_age=86400, https_only=False)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # OAuth 2.0 Protected Resource Metadata  (RFC 9728)
    # Claude.ai resolves this first from the WWW-Authenticate header, then
    # follows authorization_servers to find the authorization server.
    # ------------------------------------------------------------------

    _wk_headers = {
        "Cache-Control": "no-store, must-revalidate",
        "ETag": f'"{_SCOPE_FINGERPRINT}"',
    }

    @app.get("/.well-known/oauth-protected-resource")
    async def protected_resource_metadata(request: Request):
        base = _base_url(request)
        return JSONResponse(
            {
                "resource": base,
                "authorization_servers": [base],
                "scopes_supported": _SCOPES,
            },
            headers=_wk_headers,
        )

    # ------------------------------------------------------------------
    # OAuth 2.0 Authorization Server Metadata  (MCP spec / RFC 8414)
    # ------------------------------------------------------------------

    @app.get("/.well-known/oauth-authorization-server")
    async def oauth_metadata(request: Request):
        base = _base_url(request)
        return JSONResponse(
            {
                "issuer": base,
                "authorization_endpoint": f"{base}/authorize",
                "token_endpoint": f"{base}/token",
                "registration_endpoint": f"{base}/register",
                "response_types_supported": ["code"],
                "grant_types_supported": ["authorization_code"],
                "code_challenge_methods_supported": ["S256"],
                "token_endpoint_auth_methods_supported": ["none"],
                "scopes_supported": _SCOPES,
            },
            headers=_wk_headers,
        )

    # ------------------------------------------------------------------
    # Dynamic Client Registration  (RFC 7591)
    # ------------------------------------------------------------------

    @app.post("/register")
    async def register_client():
        return JSONResponse({
            "client_id": secrets.token_urlsafe(16),
            "client_id_issued_at": int(time.time()),
            "token_endpoint_auth_method": "none",
            "grant_types": ["authorization_code"],
            "response_types": ["code"],
        }, status_code=201)

    # ------------------------------------------------------------------
    # Authorization endpoint — proxies to Google OAuth
    # ------------------------------------------------------------------

    @app.get("/authorize")
    async def authorize(
        request: Request,
        client_id: str = "",
        redirect_uri: str = "",
        state: str = "",
        scope: str = "",
        code_challenge: str = "",
        code_challenge_method: str = "S256",
        response_type: str = "code",
    ):
        session_id = secrets.token_urlsafe(16)
        callback_uri = f"{_base_url(request)}/auth/callback"
        flow = _make_flow(callback_uri)
        auth_url, _ = flow.authorization_url(
            access_type="offline",
            prompt="consent",
            state=session_id,
        )
        _pending_auths[session_id] = {
            "client_redirect_uri": redirect_uri,
            "client_state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": code_challenge_method,
            "callback_uri": callback_uri,
            "google_code_verifier": getattr(flow, "code_verifier", None),
        }
        return RedirectResponse(auth_url)

    # ------------------------------------------------------------------
    # Token endpoint
    # ------------------------------------------------------------------

    @app.post("/token")
    async def token_endpoint(request: Request):
        form = await request.form()
        if form.get("grant_type") != "authorization_code":
            return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)

        entry = _auth_codes.pop(str(form.get("code", "")), None)
        if not entry:
            return JSONResponse({"error": "invalid_grant"}, status_code=400)
        if time.time() > entry["expires_at"]:
            return JSONResponse({"error": "invalid_grant", "error_description": "Code expired"}, status_code=400)

        if entry.get("code_challenge"):
            verifier = str(form.get("code_verifier", ""))
            if not verifier or not _verify_pkce(entry["code_challenge"], verifier):
                return JSONResponse({"error": "invalid_grant", "error_description": "PKCE failed"}, status_code=400)

        bearer = secrets.token_urlsafe(32)
        _token_store[bearer] = {"credentials": entry["credentials"], "email": entry["email"]}
        return JSONResponse({
            "access_token": bearer,
            "token_type": "bearer",
            "expires_in": 3600,
            "scope": " ".join(_SCOPES),
        })

    # ------------------------------------------------------------------
    # Google OAuth callback (shared by browser login and /authorize flow)
    # ------------------------------------------------------------------

    @app.get("/auth/callback")
    async def auth_callback(request: Request, code: str = "", state: str = "", error: str = ""):
        if error:
            return _page(f"<h1>Sign-in failed</h1><p>{html.escape(error)}</p><p><a href='/'>Try again</a></p>")

        is_mcp = state in _pending_auths
        if is_mcp:
            pending = _pending_auths.pop(state)
            callback_uri = pending["callback_uri"]
            code_verifier = pending.get("google_code_verifier")
        else:
            stored = request.session.pop("oauth_state", None)
            if state != stored:
                return _page("<h1>Invalid request</h1><p>State mismatch.</p><p><a href='/'>Try again</a></p>")
            callback_uri = request.session.pop("redirect_uri", None)
            code_verifier = request.session.pop("code_verifier", None)

        try:
            flow = _make_flow(callback_uri)
            if code_verifier:
                flow.code_verifier = code_verifier
            flow.fetch_token(code=code)
        except Exception as exc:
            logger.exception("fetch_token failed")
            return _page(f"<h1>Sign-in failed</h1><pre>{html.escape(str(exc))}</pre><p><a href='/'>Try again</a></p>")

        creds = flow.credentials
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {creds.token}"},
                timeout=10,
            )
        user_email = (resp.json() if resp.is_success else {}).get("email", "unknown")

        creds_data = {
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": list(creds.scopes) if creds.scopes else _SCOPES,
        }

        if is_mcp:
            auth_code = secrets.token_urlsafe(32)
            _auth_codes[auth_code] = {
                "credentials": creds_data,
                "email": user_email,
                "code_challenge": pending.get("code_challenge", ""),
                "code_challenge_method": pending.get("code_challenge_method", "S256"),
                "expires_at": time.time() + 600,
            }
            sep = "&" if "?" in pending["client_redirect_uri"] else "?"
            return RedirectResponse(
                f"{pending['client_redirect_uri']}{sep}code={auth_code}&state={pending['client_state']}"
            )
        else:
            bearer = secrets.token_urlsafe(32)
            _token_store[bearer] = {"credentials": creds_data, "email": user_email}
            request.session["user_email"] = user_email
            request.session["bearer_token"] = bearer
            return RedirectResponse("/")

    # ------------------------------------------------------------------
    # Browser login helpers
    # ------------------------------------------------------------------

    @app.get("/auth/google")
    async def auth_google(request: Request):
        uri = f"{_base_url(request)}/auth/callback"
        flow = _make_flow(uri)
        auth_url, state = flow.authorization_url(access_type="offline", include_granted_scopes="true", prompt="consent")
        request.session["oauth_state"] = state
        request.session["redirect_uri"] = uri
        if getattr(flow, "code_verifier", None):
            request.session["code_verifier"] = flow.code_verifier
        return RedirectResponse(auth_url)

    @app.get("/auth/logout")
    async def auth_logout(request: Request):
        _token_store.pop(request.session.get("bearer_token", ""), None)
        request.session.clear()
        return RedirectResponse("/")

    # ------------------------------------------------------------------
    # Landing page
    # ------------------------------------------------------------------

    @app.get("/")
    async def index(request: Request):
        base = _base_url(request)
        email = request.session.get("user_email")
        body = f"""\
<h1>Google Analytics MCP Server</h1>
{"<p>Signed in as <strong>" + html.escape(email) + "</strong></p>" if email else ""}
<span class="label">Add this URL to Claude.ai as a remote MCP server:</span>
<pre>{html.escape(base)}/mcp</pre>
<p>Claude.ai will open a sign-in popup automatically.</p>
{"<p><a href='/auth/logout' style='color:#888;font-size:.9rem'>Sign out</a></p>" if email else f"<br><a href='/auth/google' class='btn'>Sign in with Google</a>"}"""
        return _page(body)

    # ------------------------------------------------------------------
    # MCP endpoint (Streamable HTTP transport)
    #
    # Mounted as a raw ASGI app — NOT a FastAPI route — so that the
    # session manager writes directly to the ASGI send callable without
    # FastAPI's request_response wrapper trying to send a second response
    # on top of it (which would corrupt the stream).
    # ------------------------------------------------------------------

    class _MCPAuthApp:
        """Raw ASGI app: auth gate → session_manager."""

        async def __call__(self, scope, receive, send):
            if scope["type"] != "http":
                return

            headers = dict(scope.get("headers", []))
            session_id = headers.get(b"mcp-session-id")

            if not session_id:
                # New session — authenticate first.
                token = None

                # Check Authorization header
                auth = headers.get(b"authorization", b"").decode()
                if auth.startswith("Bearer "):
                    token = auth[7:]

                # Also accept ?token=... query param
                if not token:
                    from urllib.parse import parse_qs
                    qs = scope.get("query_string", b"").decode()
                    token = parse_qs(qs).get("token", [None])[0]

                if not token or token not in _token_store:
                    response = Response(
                        status_code=401,
                        headers={"WWW-Authenticate": f'Bearer realm="{_SCOPE_FINGERPRINT}"'},
                    )
                    await response(scope, receive, send)
                    return

                creds = _creds_from_store(_token_store[token]["credentials"])
                ctx_token = _credentials_ctx.set(creds)
                try:
                    await session_manager.handle_request(scope, receive, send)
                finally:
                    _credentials_ctx.reset(ctx_token)
            else:
                # Existing session — credentials live in the spawned server task.
                await session_manager.handle_request(scope, receive, send)

    app.mount("/mcp", _MCPAuthApp())

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_web_server():
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(create_app(), host="0.0.0.0", port=port, log_level="info")
