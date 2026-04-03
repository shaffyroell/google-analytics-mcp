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

This server implements the MCP Authorization spec so that clients like
Claude.ai can authenticate automatically via Google OAuth.

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

How Claude.ai connects
----------------------
1. Claude.ai fetches /.well-known/oauth-authorization-server to discover endpoints.
2. Claude.ai calls /register to register itself as a client.
3. Claude.ai opens a browser for the user to sign in at /authorize.
4. /authorize proxies through Google OAuth; on success redirects back to Claude.ai
   with a short-lived auth code.
5. Claude.ai calls /token to exchange the code for a bearer token.
6. Claude.ai connects to /sse with Authorization: Bearer <token>.
"""

import base64
import hashlib
import html
import logging
import os
import secrets
import time
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from mcp.server.lowlevel import NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.sse import SseServerTransport
from starlette.middleware.sessions import SessionMiddleware

import analytics_mcp.coordinator as coordinator
from analytics_mcp.tools.utils import _credentials_ctx

# Railway terminates TLS before the app sees the request.
os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
# Google returns full scope URIs even when shorthand aliases were requested.
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
# In-memory stores (replace with Redis for multi-instance deployments)
# ---------------------------------------------------------------------------

# Bearer tokens issued to MCP clients  {token -> {credentials, email}}
_token_store: Dict[str, Dict[str, Any]] = {}

# Short-lived auth codes issued after Google OAuth  {code -> {…, expires_at}}
_auth_codes: Dict[str, Dict[str, Any]] = {}

# Pending OAuth sessions initiated via /authorize
# {oauth_session_id -> {client_redirect_uri, client_state, code_challenge, …}}
_pending_auths: Dict[str, Dict[str, Any]] = {}


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


def _verify_pkce(code_challenge: str, code_verifier: str) -> bool:
    digest = hashlib.sha256(code_verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return challenge == code_challenge


# ---------------------------------------------------------------------------
# Page template
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
<body>{body}</body>
</html>"""


def _page(body: str) -> HTMLResponse:
    return HTMLResponse(_PAGE.format(body=body))


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)

    app = FastAPI(title="Google Analytics MCP Server")
    app.add_middleware(
        SessionMiddleware,
        secret_key=secret_key,
        max_age=86400,
        https_only=False,
    )

    sse = SseServerTransport("/messages/")

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # OAuth 2.0 Authorization Server Metadata (MCP spec requirement)
    # Claude.ai reads this to discover our auth endpoints.
    # ------------------------------------------------------------------

    @app.get("/.well-known/oauth-authorization-server")
    async def oauth_metadata(request: Request):
        base = _base_url(request)
        return JSONResponse({
            "issuer": base,
            "authorization_endpoint": f"{base}/authorize",
            "token_endpoint": f"{base}/token",
            "registration_endpoint": f"{base}/register",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["none"],
        })

    # ------------------------------------------------------------------
    # Dynamic Client Registration (MCP spec requirement)
    # Claude.ai registers itself here before starting the auth flow.
    # ------------------------------------------------------------------

    @app.post("/register")
    async def register_client(request: Request):
        client_id = secrets.token_urlsafe(16)
        return JSONResponse(
            {
                "client_id": client_id,
                "client_id_issued_at": int(time.time()),
                "token_endpoint_auth_method": "none",
                "grant_types": ["authorization_code"],
                "response_types": ["code"],
            },
            status_code=201,
        )

    # ------------------------------------------------------------------
    # Authorization endpoint
    # Claude.ai sends the user here to sign in. We proxy to Google OAuth
    # and, on success, redirect back to Claude.ai with our own auth code.
    # ------------------------------------------------------------------

    @app.get("/authorize")
    async def authorize(
        request: Request,
        response_type: str = "code",
        client_id: str = "",
        redirect_uri: str = "",
        state: str = "",
        scope: str = "",
        code_challenge: str = "",
        code_challenge_method: str = "S256",
    ):
        oauth_session_id = secrets.token_urlsafe(16)
        _pending_auths[oauth_session_id] = {
            "client_redirect_uri": redirect_uri,
            "client_state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": code_challenge_method,
        }

        callback_uri = f"{_base_url(request)}/auth/callback"
        flow = _make_flow(callback_uri)
        # Pass our session ID as the Google OAuth state so we can recover
        # the pending auth in the callback without relying on cookies.
        auth_url, _ = flow.authorization_url(
            access_type="offline",
            prompt="consent",
            state=oauth_session_id,
        )
        # Store the Google flow's code_verifier so the callback can use it.
        _pending_auths[oauth_session_id]["google_code_verifier"] = getattr(
            flow, "code_verifier", None
        )
        _pending_auths[oauth_session_id]["callback_uri"] = callback_uri

        return RedirectResponse(auth_url)

    # ------------------------------------------------------------------
    # Token endpoint
    # Claude.ai exchanges the auth code for a bearer token here.
    # ------------------------------------------------------------------

    @app.post("/token")
    async def token(request: Request):
        form = await request.form()
        grant_type = form.get("grant_type", "")
        code = form.get("code", "")
        code_verifier = form.get("code_verifier", "")

        if grant_type != "authorization_code":
            return JSONResponse(
                {"error": "unsupported_grant_type"}, status_code=400
            )

        entry = _auth_codes.pop(code, None)
        if not entry:
            return JSONResponse({"error": "invalid_grant"}, status_code=400)

        if time.time() > entry["expires_at"]:
            return JSONResponse(
                {"error": "invalid_grant", "error_description": "Code expired"},
                status_code=400,
            )

        # Verify PKCE if a code_challenge was stored.
        if entry.get("code_challenge"):
            if not code_verifier:
                return JSONResponse(
                    {
                        "error": "invalid_request",
                        "error_description": "code_verifier required",
                    },
                    status_code=400,
                )
            if not _verify_pkce(entry["code_challenge"], code_verifier):
                return JSONResponse(
                    {
                        "error": "invalid_grant",
                        "error_description": "PKCE verification failed",
                    },
                    status_code=400,
                )

        bearer_token = secrets.token_urlsafe(32)
        _token_store[bearer_token] = {
            "credentials": entry["credentials"],
            "email": entry["email"],
        }

        return JSONResponse({
            "access_token": bearer_token,
            "token_type": "bearer",
            "expires_in": 3600,
            "scope": " ".join(_SCOPES),
        })

    # ------------------------------------------------------------------
    # Google OAuth callback
    # Handles redirects from both /auth/google (browser) and /authorize
    # (MCP client flow).
    # ------------------------------------------------------------------

    @app.get("/auth/callback")
    async def auth_callback(
        request: Request,
        code: str = "",
        state: str = "",
        error: str = "",
    ):
        if error:
            return _page(
                f"<h1>Sign-in failed</h1><p>{html.escape(error)}</p>"
                '<p><a href="/">Try again</a></p>'
            )

        # Determine which flow triggered this callback.
        is_mcp_flow = state in _pending_auths

        if is_mcp_flow:
            pending = _pending_auths.pop(state)
            callback_uri = pending["callback_uri"]
            code_verifier = pending.get("google_code_verifier")
        else:
            # Browser login via /auth/google
            stored_state = request.session.pop("oauth_state", None)
            if state != stored_state:
                return _page(
                    "<h1>Invalid request</h1><p>State mismatch.</p>"
                    '<p><a href="/">Try again</a></p>'
                )
            callback_uri = request.session.pop("redirect_uri", None)
            code_verifier = request.session.pop("code_verifier", None)

        try:
            flow = _make_flow(callback_uri)
            if code_verifier:
                flow.code_verifier = code_verifier
            flow.fetch_token(code=code)
        except Exception as exc:
            logger.exception("fetch_token failed")
            return _page(
                f"<h1>Sign-in failed</h1><pre>{html.escape(str(exc))}</pre>"
                '<p><a href="/">Try again</a></p>'
            )

        creds = flow.credentials

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {creds.token}"},
                timeout=10,
            )
        user_info = resp.json() if resp.is_success else {}
        user_email = user_info.get("email", "unknown")

        creds_data = {
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": list(creds.scopes) if creds.scopes else _SCOPES,
        }

        if is_mcp_flow:
            # Issue a short-lived auth code and redirect back to the MCP client.
            auth_code = secrets.token_urlsafe(32)
            _auth_codes[auth_code] = {
                "credentials": creds_data,
                "email": user_email,
                "code_challenge": pending.get("code_challenge", ""),
                "code_challenge_method": pending.get(
                    "code_challenge_method", "S256"
                ),
                "expires_at": time.time() + 600,
            }
            sep = "&" if "?" in pending["client_redirect_uri"] else "?"
            redirect_url = (
                f"{pending['client_redirect_uri']}{sep}"
                f"code={auth_code}&state={pending['client_state']}"
            )
            return RedirectResponse(redirect_url)
        else:
            # Browser login: store bearer token in session and go to homepage.
            bearer_token = secrets.token_urlsafe(32)
            _token_store[bearer_token] = {
                "credentials": creds_data,
                "email": user_email,
            }
            request.session["user_email"] = user_email
            request.session["bearer_token"] = bearer_token
            return RedirectResponse("/")

    # ------------------------------------------------------------------
    # Browser-based login entry point (convenience)
    # ------------------------------------------------------------------

    @app.get("/auth/google")
    async def auth_google(request: Request):
        callback_uri = f"{_base_url(request)}/auth/callback"
        flow = _make_flow(callback_uri)
        auth_url, state = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
        )
        request.session["oauth_state"] = state
        request.session["redirect_uri"] = callback_uri
        if getattr(flow, "code_verifier", None):
            request.session["code_verifier"] = flow.code_verifier
        return RedirectResponse(auth_url)

    @app.get("/auth/logout")
    async def auth_logout(request: Request):
        token = request.session.get("bearer_token")
        if token:
            _token_store.pop(token, None)
        request.session.clear()
        return RedirectResponse("/")

    # ------------------------------------------------------------------
    # Landing page
    # ------------------------------------------------------------------

    @app.get("/")
    async def index(request: Request):
        base = _base_url(request)
        user_email = request.session.get("user_email")

        if user_email:
            safe_email = html.escape(user_email)
            body = f"""\
<h1>Google Analytics MCP Server</h1>
<p>Signed in as <strong>{safe_email}</strong></p>
<span class="label">Add this server URL to Claude.ai MCP connector:</span>
<pre>{html.escape(base)}/sse</pre>
<p>Claude.ai will handle authentication automatically.</p>
<p><a href="/auth/logout" class="logout">Sign out</a></p>"""
        else:
            body = f"""\
<h1>Google Analytics MCP Server</h1>
<p>Add this URL to Claude.ai as a remote MCP server and Claude.ai will
prompt you to sign in with Google automatically.</p>
<span class="label">MCP server URL:</span>
<pre>{html.escape(base)}/sse</pre>
<br>
<a href="/auth/google" class="btn">Sign in with Google</a>"""

        return _page(body)

    # ------------------------------------------------------------------
    # MCP SSE endpoint
    # ------------------------------------------------------------------

    @app.get("/sse")
    async def handle_sse(
        request: Request,
        token: Optional[str] = Query(default=None),
    ):
        resolved = token
        if not resolved:
            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Bearer "):
                resolved = auth_header[7:]
        if not resolved:
            resolved = request.session.get("bearer_token")

        if not resolved or resolved not in _token_store:
            # Return 401 with WWW-Authenticate so Claude.ai triggers OAuth flow.
            base = _base_url(request)
            return Response(
                status_code=401,
                headers={
                    "WWW-Authenticate": (
                        f'Bearer realm="{base}",'
                        f' resource_metadata="{base}/.well-known/oauth-authorization-server"'
                    )
                },
            )

        credentials = _credentials_from_store(
            _token_store[resolved]["credentials"]
        )
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

    app.mount("/messages/", sse.handle_post_message)

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_web_server():
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(create_app(), host="0.0.0.0", port=port, log_level="info")
