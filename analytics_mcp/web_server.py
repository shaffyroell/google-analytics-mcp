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

"""HTTP-based MCP server with Google OAuth for cloud deployment (e.g. Railway).

Setup
-----
Set these environment variables before starting:

  GOOGLE_CLIENT_ID      – OAuth 2.0 client ID from Google Cloud Console
  GOOGLE_CLIENT_SECRET  – OAuth 2.0 client secret
  SECRET_KEY            – Random secret for signing session cookies
                          (generated automatically if omitted)
  BASE_URL              – Public URL of this deployment, e.g.
                          https://my-app.up.railway.app
                          (auto-detected from request headers when omitted)
  PORT                  – TCP port to listen on (default: 8080)

Google Cloud Console setup
--------------------------
1. Create an OAuth 2.0 Web Application client.
2. Add  <BASE_URL>/auth/callback  as an authorised redirect URI.
3. Enable the Google Analytics Admin API and Analytics Data API.
"""

import html
import logging
import os
import secrets
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from mcp.server.lowlevel import NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.sse import SseServerTransport
from starlette.middleware.sessions import SessionMiddleware

import analytics_mcp.coordinator as coordinator
from analytics_mcp.tools.utils import _credentials_ctx

# Railway (and most reverse proxies) terminate TLS before the app sees the
# request, so oauthlib will reject redirect URIs as "insecure" even though
# they are HTTPS at the edge. This flag disables that transport check.
os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
# Google returns full scope URIs (e.g. https://...userinfo.email) even when
# shorthand aliases (email, profile) were requested. Relax the scope check.
os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Google OAuth scopes
# ---------------------------------------------------------------------------
_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/analytics.readonly",
]

# ---------------------------------------------------------------------------
# In-memory bearer-token store  { token -> {"credentials": {...}, "email": str} }
# For a multi-instance deployment replace this with Redis / a database.
# ---------------------------------------------------------------------------
_token_store: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Required environment variable '{name}' is not set."
        )
    return value


def _base_url(request: Request) -> str:
    """Returns the public base URL, respecting Railway / reverse-proxy headers."""
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
    return Flow.from_client_config(
        client_config, scopes=_SCOPES, redirect_uri=redirect_uri
    )


def _credentials_from_store(creds_data: dict) -> Credentials:
    return Credentials(
        token=creds_data["token"],
        refresh_token=creds_data.get("refresh_token"),
        token_uri=creds_data.get(
            "token_uri", "https://oauth2.googleapis.com/token"
        ),
        client_id=creds_data["client_id"],
        client_secret=creds_data["client_secret"],
        scopes=creds_data.get("scopes"),
    )


def _mcp_init_options() -> InitializationOptions:
    return InitializationOptions(
        server_name=coordinator.app.name,
        server_version="1.0.0",
        capabilities=coordinator.app.get_capabilities(
            notification_options=NotificationOptions(),
            experimental_capabilities={},
        ),
    )


# ---------------------------------------------------------------------------
# Page templates
# ---------------------------------------------------------------------------

_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Google Analytics MCP Server</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 700px;
            margin: 60px auto; padding: 0 20px; color: #333 }}
    h1 {{ font-size: 1.6rem; margin-bottom: .3rem }}
    pre {{ background: #f5f5f5; padding: 14px; border-radius: 6px;
           overflow-x: auto; font-size: .9rem }}
    a.btn {{ display: inline-block; padding: 10px 22px; background: #4285f4;
             color: #fff; border-radius: 5px; text-decoration: none;
             font-weight: 500 }}
    a.btn:hover {{ background: #2b6fd4 }}
    a.logout {{ color: #888; font-size: .9rem }}
    .label {{ font-weight: 600; margin-top: 1.2rem; display: block }}
  </style>
</head>
<body>
{body}
</body>
</html>"""


def _page(body: str) -> HTMLResponse:
    return HTMLResponse(_PAGE.format(body=body))


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)

    app = FastAPI(title="Google Analytics MCP Server")

    # Sessions are signed with SECRET_KEY. Use https_only=False so the cookie
    # works on Railway even when the TLS is terminated upstream.
    app.add_middleware(
        SessionMiddleware,
        secret_key=secret_key,
        max_age=86400,
        https_only=False,
    )

    sse = SseServerTransport("/messages/")

    # ------------------------------------------------------------------
    # Health check (used by Railway)
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # Landing page
    # ------------------------------------------------------------------

    @app.get("/")
    async def index(request: Request):
        user_email = request.session.get("user_email")
        bearer_token = request.session.get("bearer_token")

        if user_email and bearer_token:
            base = _base_url(request)
            sse_url = f"{base}/sse?token={bearer_token}"
            safe_email = html.escape(user_email)
            safe_token = html.escape(bearer_token)
            safe_sse = html.escape(sse_url)
            body = f"""\
<h1>Google Analytics MCP Server</h1>
<p>Signed in as <strong>{safe_email}</strong></p>

<span class="label">MCP SSE endpoint (include in your client config)</span>
<pre>{safe_sse}</pre>

<span class="label">Or configure separately</span>
<pre>SSE URL:      {html.escape(base)}/sse
Bearer token: {safe_token}</pre>

<p><a href="/auth/logout" class="logout">Sign out</a></p>"""
        else:
            body = """\
<h1>Google Analytics MCP Server</h1>
<p>Sign in with your Google account to get an MCP access token
for your Google Analytics data.</p>
<br>
<a href="/auth/google" class="btn">Sign in with Google</a>"""

        return _page(body)

    # ------------------------------------------------------------------
    # OAuth: start
    # ------------------------------------------------------------------

    @app.get("/auth/google")
    async def auth_google(request: Request):
        redirect_uri = f"{_base_url(request)}/auth/callback"
        flow = _make_flow(redirect_uri)
        auth_url, state = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
        )
        request.session["oauth_state"] = state
        request.session["redirect_uri"] = redirect_uri
        # Persist the PKCE code verifier so the callback can send it back to
        # Google when exchanging the code (google-auth-oauthlib generates this
        # automatically; the callback creates a new Flow so it must be restored).
        if getattr(flow, "code_verifier", None):
            request.session["code_verifier"] = flow.code_verifier
        return RedirectResponse(auth_url)

    # ------------------------------------------------------------------
    # OAuth: callback
    # ------------------------------------------------------------------

    @app.get("/auth/callback")
    async def auth_callback(
        request: Request,
        code: str = "",
        state: str = "",
        error: str = "",
    ):
        if error:
            safe_err = html.escape(error)
            return _page(
                f"<h1>Sign-in failed</h1><p>{safe_err}</p>"
                '<p><a href="/">Try again</a></p>'
            )

        stored_state = request.session.pop("oauth_state", None)
        redirect_uri = request.session.pop("redirect_uri", None)

        if state != stored_state:
            return _page(
                "<h1>Invalid request</h1><p>OAuth state mismatch.</p>"
                '<p><a href="/">Try again</a></p>',
            )

        code_verifier = request.session.pop("code_verifier", None)

        try:
            flow = _make_flow(redirect_uri)
            if code_verifier:
                flow.code_verifier = code_verifier
            flow.fetch_token(code=code)
        except Exception as exc:
            logger.exception("fetch_token failed")
            safe_exc = html.escape(str(exc))
            return _page(
                f"<h1>Sign-in failed</h1><pre>{safe_exc}</pre>"
                '<p><a href="/">Try again</a></p>'
            )

        creds = flow.credentials

        # Fetch the user's email address.
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {creds.token}"},
                timeout=10,
            )
        user_info = resp.json() if resp.is_success else {}
        user_email = user_info.get("email", "unknown")

        bearer_token = secrets.token_urlsafe(32)
        _token_store[bearer_token] = {
            "credentials": {
                "token": creds.token,
                "refresh_token": creds.refresh_token,
                "token_uri": creds.token_uri,
                "client_id": creds.client_id,
                "client_secret": creds.client_secret,
                "scopes": list(creds.scopes) if creds.scopes else _SCOPES,
            },
            "email": user_email,
        }

        request.session["user_email"] = user_email
        request.session["bearer_token"] = bearer_token

        return RedirectResponse("/")

    # ------------------------------------------------------------------
    # OAuth: logout
    # ------------------------------------------------------------------

    @app.get("/auth/logout")
    async def auth_logout(request: Request):
        token = request.session.get("bearer_token")
        if token:
            _token_store.pop(token, None)
        request.session.clear()
        return RedirectResponse("/")

    # ------------------------------------------------------------------
    # MCP SSE endpoint
    # ------------------------------------------------------------------

    @app.get("/sse")
    async def handle_sse(
        request: Request,
        token: Optional[str] = Query(default=None),
    ):
        # Accept token from:
        #   1. ?token= query param  (easiest for MCP client config URLs)
        #   2. Authorization: Bearer <token> header
        #   3. Session cookie (set after browser login)
        resolved = token
        if not resolved:
            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Bearer "):
                resolved = auth_header[7:]
        if not resolved:
            resolved = request.session.get("bearer_token")

        if not resolved or resolved not in _token_store:
            return _page(
                "<h1>Unauthorised</h1>"
                "<p>Please <a href='/auth/google'>sign in</a> first.</p>"
            )

        credentials = _credentials_from_store(
            _token_store[resolved]["credentials"]
        )

        # Store credentials in the async context so all tool calls in this
        # MCP session use this user's Google credentials.
        ctx_token = _credentials_ctx.set(credentials)
        try:
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await coordinator.app.run(
                    streams[0], streams[1], _mcp_init_options()
                )
        finally:
            _credentials_ctx.reset(ctx_token)

    # Mount the SSE message handler (client POSTs tool calls here).
    app.mount("/messages/", sse.handle_post_message)

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_web_server():
    """Start the web server (used by the analytics-mcp-web CLI entry point)."""
    import uvicorn

    # Allow HTTP during local development without triggering oauthlib's
    # HTTPS-only guard.
    if os.environ.get("ENV", "production").lower() == "development":
        os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(create_app(), host="0.0.0.0", port=port, log_level="info")
