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
  SECRET_KEY            – Encrypts the tokens issued to Claude and signs
                          cookies. Set it to a long random value and never
                          change it, or every connected client is signed
                          out. If unset it is derived from
                          GOOGLE_CLIENT_SECRET.
  BASE_URL              – Public URL, e.g. https://my-app.up.railway.app
  PORT                  – Listen port (default: 8080)

Add  <BASE_URL>/auth/callback  as an authorised redirect URI in Google
Cloud Console.

Staying signed in
-----------------
Everything Claude holds is self-contained, so nothing is lost on a restart
or redeploy: client registrations are signed into the client_id, and access
and refresh tokens are encrypted payloads carrying the user's Google
credentials. Claude refreshes silently; users only reconnect if they revoke
access at Google — or if the Google OAuth app is left in "Testing", where
Google expires refresh tokens after 7 days.
"""

import base64
import contextlib
import hashlib
import hmac
import html
import json
import logging
import os
import secrets
import time
from datetime import datetime, timezone
from typing import Any

import anyio
import httpx
from cryptography.fernet import Fernet, InvalidToken
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    RefreshToken,
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
from starlette.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from starlette.routing import Route
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
    return Flow.from_client_config(
        client_config, scopes=_SCOPES, redirect_uri=redirect_uri
    )


def _creds_from_store(data: dict) -> Credentials:
    """Rebuild Google credentials from a token payload.

    The OAuth client id/secret come from the environment. ``expiry`` must be
    set, or google-auth treats the ~1h Google access token as valid forever
    and never refreshes it; a missing expiry is treated as already expired.
    """
    expiry_ts = data.get("expiry")
    expiry = (
        datetime.fromtimestamp(expiry_ts, timezone.utc).replace(tzinfo=None)
        if expiry_ts
        else datetime(1970, 1, 1)
    )
    creds = Credentials(
        token=data["token"],
        refresh_token=data.get("refresh_token"),
        token_uri=data.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=os.environ.get("GOOGLE_CLIENT_ID"),
        client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
        scopes=data.get("scopes"),
    )
    creds.expiry = expiry
    return creds


def _creds_to_store(creds: Credentials) -> dict:
    """Serialise Google credentials for embedding in an encrypted token.

    The OAuth client id/secret are deliberately left out — they live in env.
    """
    expiry = None
    if creds.expiry:
        expiry = int(creds.expiry.replace(tzinfo=timezone.utc).timestamp())
    return {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "scopes": list(creds.scopes) if creds.scopes else _SCOPES,
        "expiry": expiry,
    }


# Refreshed Google access tokens, keyed by a hash of the Google refresh
# token. Bearer tokens are immutable, so without this every request made
# more than an hour after the bearer was minted would hit Google's token
# endpoint. Purely an optimisation: losing it on restart costs one refresh.
_google_token_cache: dict[str, tuple[str, datetime]] = {}


def _cache_key(creds: Credentials) -> str | None:
    if not creds.refresh_token:
        return None
    return hashlib.sha256(creds.refresh_token.encode()).hexdigest()


def _apply_cached_google_token(creds: Credentials) -> None:
    key = _cache_key(creds)
    cached = _google_token_cache.get(key) if key else None
    if cached and (creds.expiry is None or cached[1] > creds.expiry):
        creds.token, creds.expiry = cached


def _remember_google_token(creds: Credentials) -> None:
    key = _cache_key(creds)
    if key and creds.token and creds.expiry:
        _google_token_cache[key] = (creds.token, creds.expiry)


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

_ACCESS_TOKEN_TTL = 8 * 3600  # 8 hours; Claude refreshes silently
_REFRESH_TOKEN_TTL = 365 * 24 * 3600  # 1 year, renewed on every refresh
_CLIENT_ID_PREFIX = "mcpc_"


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _resolve_secret_key() -> str:
    explicit = os.environ.get("SECRET_KEY")
    if explicit:
        return explicit
    google_secret = os.environ.get("GOOGLE_CLIENT_SECRET")
    if google_secret:
        logger.warning(
            "SECRET_KEY is not set; deriving one from GOOGLE_CLIENT_SECRET so "
            "tokens survive restarts. Set SECRET_KEY explicitly for "
            "production."
        )
        return hashlib.sha256(
            b"analytics-mcp/secret-key:" + google_secret.encode()
        ).hexdigest()
    logger.warning(
        "SECRET_KEY is not set and there is no GOOGLE_CLIENT_SECRET to derive "
        "one from: using a random key. Every restart will sign clients out."
    )
    return secrets.token_hex(32)


class _TokenCodec:
    """Encrypts/decrypts token payloads and signs client IDs.

    Tokens are Fernet-encrypted (AES + HMAC) because they carry the user's
    Google refresh token. Client IDs are only signed: they are not secret.
    """

    def __init__(self, secret: str) -> None:
        self._secret = secret
        fernet_key = base64.urlsafe_b64encode(
            hashlib.sha256(b"analytics-mcp/fernet:" + secret.encode()).digest()
        )
        self._fernet = Fernet(fernet_key)

    def _mac(self, purpose: str, data: str) -> str:
        return hmac.new(
            self._secret.encode(),
            f"{purpose}:{data}".encode(),
            hashlib.sha256,
        ).hexdigest()

    def encode(self, payload: dict) -> str:
        return self._fernet.encrypt(json.dumps(payload).encode()).decode()

    def decode(self, token: str, typ: str) -> dict | None:
        """Return the payload if ``token`` is a valid, unexpired ``typ``."""
        try:
            payload = json.loads(self._fernet.decrypt(token.encode()))
        except (InvalidToken, ValueError, TypeError):
            return None
        if payload.get("typ") != typ:
            return None
        if time.time() > payload.get("expires_at", 0):
            return None
        return payload

    def encode_client_id(self, meta: dict) -> str:
        body = _b64e(json.dumps(meta, separators=(",", ":")).encode())
        sig = self._mac("client-id", body)[:32]
        return f"{_CLIENT_ID_PREFIX}{body}.{sig}"

    def decode_client_id(self, client_id: str) -> dict | None:
        if not client_id.startswith(_CLIENT_ID_PREFIX):
            return None
        try:
            body, sig = client_id[len(_CLIENT_ID_PREFIX) :].rsplit(".", 1)
            if not hmac.compare_digest(sig, self._mac("client-id", body)[:32]):
                return None
            return json.loads(_b64d(body))
        except Exception:
            return None

    def client_secret_for(self, client_id: str) -> str:
        return self._mac("client-secret", client_id)


class GoogleOAuthProvider(OAuthAuthorizationServerProvider):
    """MCP OAuth AS that proxies authentication to Google OAuth 2.0.

    Stateless across restarts: client registrations, access tokens and
    refresh tokens are all self-contained. Only the short-lived (~minutes)
    login handshake state lives in memory.
    """

    def __init__(self, secret_key: str) -> None:
        self._codec = _TokenCodec(secret_key)
        # MCP auth codes issued after Google callback
        self._auth_codes: dict[str, AuthorizationCode] = {}
        # Google credentials associated with each MCP auth code
        self._auth_code_creds: dict[str, dict] = {}
        # Pending Google OAuth flows keyed by google_state
        self._pending: dict[str, dict[str, Any]] = {}

    # --- clients -----------------------------------------------------------

    def new_client(
        self,
        redirect_uris: list[str],
        auth_method: str,
        grant_types: list[str],
        response_types: list[str],
    ) -> OAuthClientInformationFull:
        client_id = self._codec.encode_client_id(
            {
                "r": redirect_uris,
                "m": auth_method,
                "g": grant_types,
                "t": response_types,
                "i": int(time.time()),
            }
        )
        meta = self._codec.decode_client_id(client_id)
        return self._client_from_meta(client_id, meta)

    def _client_from_meta(
        self, client_id: str, meta: dict
    ) -> OAuthClientInformationFull:
        auth_method = meta["m"]
        # Pass redirect_uris as plain strings so Pydantic converts them to
        # AnyUrl (the field type on OAuthClientInformationFull); AnyHttpUrl
        # would make validate_redirect_uri() comparisons fail silently.
        return OAuthClientInformationFull(
            client_id=client_id,
            client_secret=(
                self._codec.client_secret_for(client_id)
                if auth_method != "none"
                else None
            ),
            client_id_issued_at=meta["i"],
            redirect_uris=meta["r"],
            token_endpoint_auth_method=auth_method,
            grant_types=meta["g"],
            response_types=meta["t"],
            scope=" ".join(_SCOPES),
        )

    async def get_client(
        self, client_id: str
    ) -> OAuthClientInformationFull | None:
        meta = self._codec.decode_client_id(client_id)
        if meta is None:
            return None
        return self._client_from_meta(client_id, meta)

    async def register_client(
        self, client_info: OAuthClientInformationFull
    ) -> None:
        # Registration goes through the custom /register route, which mints
        # stateless client ids via new_client(); nothing to store.
        pass

    # --- tokens ------------------------------------------------------------

    def _issue_tokens(
        self, creds_data: dict, email: str, client_id: str, scopes: list[str]
    ) -> OAuthToken:
        now = int(time.time())
        common = {
            "credentials": creds_data,
            "email": email,
            "client_id": client_id,
            "scopes": list(scopes),
        }
        access = self._codec.encode(
            {**common, "typ": "access", "expires_at": now + _ACCESS_TOKEN_TTL}
        )
        refresh = self._codec.encode(
            {
                **common,
                "typ": "refresh",
                "expires_at": now + _REFRESH_TOKEN_TTL,
            }
        )
        return OAuthToken(
            access_token=access,
            token_type="Bearer",
            expires_in=_ACCESS_TOKEN_TTL,
            scope=" ".join(scopes),
            refresh_token=refresh,
        )

    def browser_session_token(self, creds_data: dict, email: str) -> str:
        return self._codec.encode(
            {
                "credentials": creds_data,
                "email": email,
                "client_id": "browser",
                "scopes": _SCOPES,
                "typ": "access",
                "expires_at": int(time.time()) + 86400,
            }
        )

    def decode_access_token(self, token: str) -> dict | None:
        return self._codec.decode(token, typ="access")

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
        self,
        client: OAuthClientInformationFull,
        authorization_code: AuthorizationCode,
    ) -> OAuthToken:
        creds_data = self._auth_code_creds.pop(authorization_code.code, None)
        if creds_data is None:
            raise TokenError(
                error="invalid_grant", error_description="Credentials not found"
            )

        self._auth_codes.pop(authorization_code.code, None)

        return self._issue_tokens(
            creds_data["credentials"],
            creds_data["email"],
            client.client_id,
            list(authorization_code.scopes),
        )

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        payload = self._codec.decode(refresh_token, typ="refresh")
        if payload is None:
            return None
        return RefreshToken(
            token=refresh_token,
            client_id=payload["client_id"],
            scopes=payload["scopes"],
            expires_at=payload["expires_at"],
        )

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        payload = self._codec.decode(refresh_token.token, typ="refresh")
        if payload is None:
            raise TokenError(
                error="invalid_grant",
                error_description="refresh token is invalid",
            )

        # Refresh the Google access token now, so the new bearer starts with
        # a fresh one and a revoked/expired Google grant surfaces here as
        # invalid_grant (Claude then asks the user to reconnect) rather than
        # as confusing tool errors later.
        creds = _creds_from_store(payload["credentials"])
        try:
            await anyio.to_thread.run_sync(creds.refresh, GoogleAuthRequest())
        except RefreshError as exc:
            logger.warning(
                "Google refresh failed for %s: %s", payload.get("email"), exc
            )
            raise TokenError(
                error="invalid_grant",
                error_description=(
                    "Google authorization expired or was revoked; "
                    "please reconnect"
                ),
            ) from exc
        except Exception:
            # Transient (network, Google 5xx): don't sign the user out over
            # it. Keep the old Google token; it is refreshed on first use.
            logger.exception(
                "Transient error refreshing Google token; keeping existing "
                "credentials"
            )
        else:
            _remember_google_token(creds)
            # Google normally omits refresh_token on refresh; google-auth
            # keeps the existing one on the credentials object.
            payload["credentials"] = _creds_to_store(creds)

        logger.info("Refreshed MCP tokens for %s", payload.get("email"))
        return self._issue_tokens(
            payload["credentials"], payload["email"], client.client_id, scopes
        )

    async def load_access_token(self, token: str) -> AccessToken | None:
        payload = self.decode_access_token(token)
        if payload is None:
            return None
        return AccessToken(
            token=token,
            client_id=payload["client_id"],
            scopes=payload["scopes"],
            expires_at=payload["expires_at"],
        )

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        pass  # stateless tokens cannot be revoked; they expire naturally


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> Starlette:
    secret_key = _resolve_secret_key()
    base_url = os.environ.get("BASE_URL", "").rstrip("/")

    provider = GoogleOAuthProvider(secret_key)

    # Stateless: every request is authenticated and carries its own
    # credentials, so nothing breaks when Railway restarts the process.
    session_manager = StreamableHTTPSessionManager(
        app=coordinator.app,
        json_response=False,
        stateless=True,
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
    # Override /.well-known/oauth-authorization-server to include "none"
    # in token_endpoint_auth_methods_supported. The SDK's built-in metadata
    # only advertises client_secret_post and client_secret_basic; Claude.ai
    # is a public client that uses "none", so we must advertise it.
    # ------------------------------------------------------------------

    async def oauth_server_metadata(request: Request) -> JSONResponse:
        base = _get_base_url(request)
        metadata = {
            "issuer": base,
            "authorization_endpoint": f"{base}/authorize",
            "token_endpoint": f"{base}/token",
            "registration_endpoint": f"{base}/register",
            "scopes_supported": _SCOPES,
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "token_endpoint_auth_methods_supported": [
                "none",
                "client_secret_post",
                "client_secret_basic",
            ],
            "code_challenge_methods_supported": ["S256"],
        }
        return JSONResponse(metadata, headers={"Cache-Control": "no-store"})

    # ------------------------------------------------------------------
    # Custom /register — must come BEFORE auth_routes in the route list
    # so it takes priority over the SDK's built-in handler.
    #
    # The SDK's handler (register.py:76) rejects registrations where
    # grant_types doesn't include BOTH authorization_code AND refresh_token.
    # Claude.ai registers with only ["authorization_code"] per the MCP spec
    # for public clients, which causes the SDK to return 400 and Claude.ai
    # shows "Authorization failed". We normalise grant_types here.
    # ------------------------------------------------------------------

    async def register_client(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                {"error": "invalid_client_metadata"}, status_code=400
            )

        logger.info("CLIENT REGISTRATION: body=%s", body)
        grant_types: list = body.get(
            "grant_types", ["authorization_code", "refresh_token"]
        )
        # Add refresh_token so the SDK's /authorize + /token handlers are happy.
        if "authorization_code" not in grant_types:
            return JSONResponse(
                {
                    "error": "invalid_client_metadata",
                    "error_description": "grant_types must include authorization_code",
                },
                status_code=400,
            )
        if "refresh_token" not in grant_types:
            grant_types = list(grant_types) + ["refresh_token"]

        redirect_uris_raw: list = body.get("redirect_uris", [])
        if not redirect_uris_raw:
            return JSONResponse(
                {
                    "error": "invalid_client_metadata",
                    "error_description": "redirect_uris is required",
                },
                status_code=400,
            )

        auth_method: str = body.get("token_endpoint_auth_method") or "none"
        if auth_method not in (
            "none",
            "client_secret_post",
            "client_secret_basic",
        ):
            return JSONResponse(
                {
                    "error": "invalid_client_metadata",
                    "error_description": (
                        "unsupported token_endpoint_auth_method "
                        f"{auth_method}"
                    ),
                },
                status_code=400,
            )

        # Stateless client id: the registration is encoded (and signed) in
        # the id itself, so it is still valid after a restart or redeploy.
        client_info = provider.new_client(
            redirect_uris=[str(u) for u in redirect_uris_raw],
            auth_method=auth_method,
            grant_types=list(grant_types),
            response_types=body.get("response_types", ["code"]),
        )
        client_id = client_info.client_id
        client_secret = client_info.client_secret
        logger.info(
            "CLIENT REGISTERED: auth_method=%s redirect_uris=%s",
            auth_method,
            redirect_uris_raw,
        )

        resp_body: dict = {
            "client_id": client_id,
            "client_id_issued_at": client_info.client_id_issued_at,
            "redirect_uris": redirect_uris_raw,
            "grant_types": grant_types,
            "response_types": ["code"],
            "token_endpoint_auth_method": auth_method,
            "scope": " ".join(_SCOPES),
        }
        if client_secret:
            resp_body["client_secret"] = client_secret
        return JSONResponse(resp_body, status_code=201)

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
            {
                "resource": base,
                "authorization_servers": [base],
                "scopes_supported": _SCOPES,
            },
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
            return _page(
                f"<h1>Sign-in failed</h1><p>{html.escape(error)}</p><p><a href='/'>Try again</a></p>"
            )

        is_mcp = state in provider._pending

        if is_mcp:
            pending = provider._pending.pop(state)
            callback_uri = pending["callback_uri"]
            code_verifier = pending.get("code_verifier")
        else:
            stored = request.session.pop("oauth_state", None)
            if state != stored:
                return _page(
                    "<h1>Invalid request</h1><p>State mismatch.</p><p><a href='/'>Try again</a></p>"
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
                f"<h1>Sign-in failed</h1><pre>{html.escape(str(exc))}</pre><p><a href='/'>Try again</a></p>"
            )

        creds = flow.credentials
        user_email = await _fetch_email(creds.token)

        if not creds.refresh_token:
            logger.warning(
                "Google returned no refresh token for %s; the connection will "
                "only last until the Google access token expires (~1h)",
                user_email,
            )
        creds_data = _creds_to_store(creds)

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
            bearer = provider.browser_session_token(creds_data, user_email)
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
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
        )
        request.session["oauth_state"] = state
        request.session["redirect_uri"] = uri
        if getattr(flow, "code_verifier", None):
            request.session["code_verifier"] = flow.code_verifier
        return RedirectResponse(auth_url)

    async def auth_logout(request: Request) -> Response:
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
            + (
                f"<p>Signed in as <strong>{html.escape(email)}</strong></p>"
                if email
                else ""
            )
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
        async def __call__(
            self, scope: Scope, receive: Receive, send: Send
        ) -> None:
            if scope["type"] != "http":
                return

            headers = dict(scope.get("headers", []))
            token: str | None = None

            auth = headers.get(b"authorization", b"").decode()
            if auth.startswith("Bearer "):
                token = auth[7:]

            if not token:
                from urllib.parse import parse_qs

                qs = scope.get("query_string", b"").decode()
                token = (parse_qs(qs).get("token") or [None])[0]

            # Every request is authenticated (sessions are stateless).
            payload = provider.decode_access_token(token) if token else None
            if not payload:
                logger.warning(
                    "MCP /mcp: 401 — token=%s valid=%s",
                    bool(token),
                    bool(payload),
                )
                resp = Response(
                    status_code=401,
                    headers={
                        "WWW-Authenticate": (
                            f'Bearer realm="{_SCOPE_FINGERPRINT}"'
                        )
                    },
                )
                await resp(scope, receive, send)
                return

            creds = _creds_from_store(payload["credentials"])
            _apply_cached_google_token(creds)
            ctx_token = _credentials_ctx.set(creds)
            try:
                await session_manager.handle_request(scope, receive, send)
            finally:
                _credentials_ctx.reset(ctx_token)
                _remember_google_token(creds)

    # ------------------------------------------------------------------
    # Assemble Starlette app
    # ------------------------------------------------------------------

    routes = [
        # These MUST come before auth_routes to shadow the SDK's built-in handlers.
        Route(
            "/.well-known/oauth-authorization-server",
            oauth_server_metadata,
            methods=["GET"],
        ),
        Route("/register", register_client, methods=["POST"]),
        *auth_routes,  # /authorize, /token (/.well-known/oauth-authorization-server and /register shadowed above)
        Route(
            "/.well-known/oauth-protected-resource", protected_resource_metadata
        ),
        Route("/auth/callback", auth_callback),
        Route("/auth/google", auth_google),
        Route("/auth/logout", auth_logout),
        Route("/health", health),
        Route("/", index),
        Route("/mcp", _MCPAuthApp()),
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
    app.state.provider = provider
    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_web_server() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(create_app(), host="0.0.0.0", port=port, log_level="info")
