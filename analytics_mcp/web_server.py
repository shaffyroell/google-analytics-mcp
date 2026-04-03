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

Uses the MCP SDK's built-in auth infrastructure (mcp.server.auth.*) so the
OAuth endpoints are spec-correct and compatible with Claude.ai's MCP connector.
Google OAuth is used as the identity provider — we act as an OAuth AS that
proxies authentication to Google.

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
from typing import Any

import httpx
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    TokenError,
)
from mcp.server.auth.routes import create_auth_routes
from mcp.server.auth.settings import ClientRegistrationOptions
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import AnyHttpUrl
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Mount, Route
from starlette.types import Receive, Scope, Send

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

# 12-char hex fingerprint of sorted scopes — used as Bearer realm value and
# ETag on well-known responses (matches google_workspace_mcp pattern).
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


def _get_base_url(request: Request) -> str:
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


async def _fetch_email(access_token: str) -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
    return (resp.json() if resp.is_success else {}).get("email", "unknown")


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
# OAuth provider — proxies identity to Google
# ---------------------------------------------------------------------------

class GoogleOAuthProvider(OAuthAuthorizationServerProvider):
    """MCP OAuth AS that proxies authentication to Google OAuth 2.0."""

    def __init__(self) -> None:
        # Registered MCP clients (dynamic registration)
        self._clients: dict[str, OAuthClientInformationFull] = {}
        # MCP auth codes issued after Google callback
        self._auth_codes: dict[str, AuthorizationCode] = {}
        # Google credentials associated with each MCP auth code
        self._auth_code_creds: dict[str, dict] = {}
        # Issued bearer tokens → credentials + metadata
        self._tokens: dict[str, dict[str, Any]] = {}
        # Pending Google OAuth flows keyed by google_state
        self._pending: dict[str, dict[str, Any]] = {}

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self._clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        self._clients[client_info.client_id] = client_info

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        """Start Google OAuth flow and return the Google authorization URL."""
        base_url = os.environ.get("BASE_URL", "").rstrip("/")
        if not base_url:
            raise RuntimeError("BASE_URL env var is required for MCP OAuth")

        google_state = secrets.token_urlsafe(16)
        callback_uri = f"{base_url}/auth/callback"
        flow = _make_flow(callback_uri)
        auth_url, _ = flow.authorization_url(
            access_type="offline",
            prompt="consent",
            state=google_state,
        )
        self._pending[google_state] = {
            "params": params,
            "client": client,
            "code_verifier": getattr(flow, "code_verifier", None),
            "callback_uri": callback_uri,
        }
        return auth_url

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        return self._auth_codes.get(authorization_code)

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        creds_data = self._auth_code_creds.pop(authorization_code.code, None)
        if creds_data is None:
            raise TokenError(error="invalid_grant", error_description="Credentials not found")

        del self._auth_codes[authorization_code.code]

        bearer = secrets.token_urlsafe(32)
        self._tokens[bearer] = {
            "credentials": creds_data["credentials"],
            "email": creds_data["email"],
            "client_id": client.client_id,
            "scopes": authorization_code.scopes,
            "expires_at": int(time.time()) + 3600,
        }
        return OAuthToken(
            access_token=bearer,
            token_type="Bearer",
            expires_in=3600,
            scope=" ".join(authorization_code.scopes),
        )

    async def load_refresh_token(self, client: OAuthClientInformationFull, refresh_token: str):
        return None  # Refresh tokens not supported

    async def exchange_refresh_token(self, client, refresh_token, scopes):
        raise TokenError(error="unsupported_grant_type")

    async def load_access_token(self, token: str) -> AccessToken | None:
        entry = self._tokens.get(token)
        if not entry:
            return None
        if time.time() > entry["expires_at"]:
            return None
        return AccessToken(
            token=token,
            client_id=entry["client_id"],
            scopes=entry["scopes"],
            expires_at=entry["expires_at"],
        )

    async def revoke_token(self, token: AccessToken) -> None:
        self._tokens.pop(token.token, None)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> Starlette:
    secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
    base_url = os.environ.get("BASE_URL", "").rstrip("/")

    provider = GoogleOAuthProvider()

    session_manager = StreamableHTTPSessionManager(
        app=coordinator.app,
        json_response=False,
        stateless=False,
        session_idle_timeout=1800,
    )

    # ------------------------------------------------------------------
    # SDK OAuth routes — handles /.well-known/oauth-authorization-server,
    # /register, /authorize, /token correctly per spec.
    # ------------------------------------------------------------------

    # issuer_url must be a valid AnyHttpUrl; fall back to a placeholder if
    # BASE_URL isn't set yet (it will be set in the Railway environment).
    issuer = base_url or "https://placeholder.invalid"
    auth_routes = create_auth_routes(
        provider=provider,
        issuer_url=AnyHttpUrl(issuer),
        client_registration_options=ClientRegistrationOptions(
            enabled=True,
            valid_scopes=_SCOPES,
            default_scopes=_SCOPES,
        ),
    )

    # ------------------------------------------------------------------
    # Protected Resource Metadata (RFC 9728)
    # ------------------------------------------------------------------

    _wk_headers = {
        "Cache-Control": "no-store, must-revalidate",
        "ETag": f'"{_SCOPE_FINGERPRINT}"',
    }

    async def protected_resource_metadata(request: Request) -> JSONResponse:
        base = _get_base_url(request)
        return JSONResponse(
            {"resource": base, "authorization_servers": [base], "scopes_supported": _SCOPES},
            headers=_wk_headers,
        )

    # ------------------------------------------------------------------
    # Google OAuth callback — shared by MCP flow and browser login
    # ------------------------------------------------------------------

    async def auth_callback(request: Request) -> Response:
        code = request.query_params.get("code", "")
        state = request.query_params.get("state", "")
        error = request.query_params.get("error", "")

        if error:
            return _page(f"<h1>Sign-in failed</h1><p>{html.escape(error)}</p><p><a href='/'>Try again</a></p>")

        is_mcp = state in provider._pending

        if is_mcp:
            pending = provider._pending.pop(state)
            callback_uri = pending["callback_uri"]
            code_verifier = pending.get("code_verifier")
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
        user_email = await _fetch_email(creds.token)

        creds_data = {
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": list(creds.scopes) if creds.scopes else _SCOPES,
        }

        if is_mcp:
            params: AuthorizationParams = pending["params"]
            client: OAuthClientInformationFull = pending["client"]
            mcp_code = secrets.token_urlsafe(32)

            auth_code_obj = AuthorizationCode(
                code=mcp_code,
                scopes=params.scopes or _SCOPES,
                expires_at=time.time() + 600,
                client_id=client.client_id,
                code_challenge=params.code_challenge,
                redirect_uri=params.redirect_uri,
                redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            )
            provider._auth_codes[mcp_code] = auth_code_obj
            provider._auth_code_creds[mcp_code] = {
                "credentials": creds_data,
                "email": user_email,
            }

            redirect_uri_str = str(params.redirect_uri)
            sep = "&" if "?" in redirect_uri_str else "?"
            return RedirectResponse(
                f"{redirect_uri_str}{sep}code={mcp_code}&state={params.state or ''}",
                status_code=302,
            )
        else:
            bearer = secrets.token_urlsafe(32)
            provider._tokens[bearer] = {
                "credentials": creds_data,
                "email": user_email,
                "client_id": "browser",
                "scopes": _SCOPES,
                "expires_at": int(time.time()) + 86400,
            }
            request.session["user_email"] = user_email
            request.session["bearer_token"] = bearer
            return RedirectResponse("/", status_code=302)

    # ------------------------------------------------------------------
    # Browser login helpers
    # ------------------------------------------------------------------

    async def auth_google(request: Request) -> Response:
        base = _get_base_url(request)
        uri = f"{base}/auth/callback"
        flow = _make_flow(uri)
        auth_url, state = flow.authorization_url(
            access_type="offline", include_granted_scopes="true", prompt="consent"
        )
        request.session["oauth_state"] = state
        request.session["redirect_uri"] = uri
        if getattr(flow, "code_verifier", None):
            request.session["code_verifier"] = flow.code_verifier
        return RedirectResponse(auth_url)

    async def auth_logout(request: Request) -> Response:
        provider._tokens.pop(request.session.get("bearer_token", ""), None)
        request.session.clear()
        return RedirectResponse("/")

    # ------------------------------------------------------------------
    # Landing page and health
    # ------------------------------------------------------------------

    async def health(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def index(request: Request) -> HTMLResponse:
        base = _get_base_url(request)
        email = request.session.get("user_email")
        body = (
            f"<h1>Google Analytics MCP Server</h1>"
            + (f"<p>Signed in as <strong>{html.escape(email)}</strong></p>" if email else "")
            + f'<span class="label">Add this URL to Claude.ai as a remote MCP server:</span>'
            + f"<pre>{html.escape(base)}/mcp</pre>"
            + "<p>Claude.ai will open a sign-in popup automatically.</p>"
            + (
                "<p><a href='/auth/logout' style='color:#888;font-size:.9rem'>Sign out</a></p>"
                if email
                else f"<br><a href='/auth/google' class='btn'>Sign in with Google</a>"
            )
        )
        return _page(body)

    # ------------------------------------------------------------------
    # MCP endpoint — raw ASGI app mounted at /mcp
    #
    # Mounted as a plain ASGI callable (NOT a Starlette/FastAPI route) so
    # that session_manager.handle_request() writes directly to the ASGI
    # send callable with no response-wrapper interference.
    # ------------------------------------------------------------------

    class _MCPAuthApp:
        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] != "http":
                return

            headers = dict(scope.get("headers", []))
            session_id = headers.get(b"mcp-session-id")

            if not session_id:
                # New session — validate Bearer token first.
                token: str | None = None

                auth = headers.get(b"authorization", b"").decode()
                if auth.startswith("Bearer "):
                    token = auth[7:]

                if not token:
                    from urllib.parse import parse_qs
                    qs = scope.get("query_string", b"").decode()
                    token = (parse_qs(qs).get("token") or [None])[0]

                # Validate via provider (checks existence and expiry).
                access_token_obj = None
                if token:
                    access_token_obj = await provider.load_access_token(token)

                if not access_token_obj or not token or token not in provider._tokens:
                    resp = Response(
                        status_code=401,
                        headers={"WWW-Authenticate": f'Bearer realm="{_SCOPE_FINGERPRINT}"'},
                    )
                    await resp(scope, receive, send)
                    return

                creds = _creds_from_store(provider._tokens[token]["credentials"])
                ctx_token = _credentials_ctx.set(creds)
                try:
                    await session_manager.handle_request(scope, receive, send)
                finally:
                    _credentials_ctx.reset(ctx_token)
            else:
                # Existing session — credentials live in the spawned server task.
                await session_manager.handle_request(scope, receive, send)

    # ------------------------------------------------------------------
    # Assemble Starlette app
    # ------------------------------------------------------------------

    routes = [
        *auth_routes,  # /.well-known/oauth-authorization-server, /register, /authorize, /token
        Route("/.well-known/oauth-protected-resource", protected_resource_metadata),
        Route("/auth/callback", auth_callback),
        Route("/auth/google", auth_google),
        Route("/auth/logout", auth_logout),
        Route("/health", health),
        Route("/", index),
        Mount("/mcp", app=_MCPAuthApp()),
    ]

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        async with session_manager.run():
            yield

    app = Starlette(
        routes=routes,
        middleware=[
            Middleware(
                SessionMiddleware,
                secret_key=secret_key,
                max_age=86400,
                https_only=False,
            )
        ],
        lifespan=lifespan,
    )
    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_web_server() -> None:
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(create_app(), host="0.0.0.0", port=port, log_level="info")
