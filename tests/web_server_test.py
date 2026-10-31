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

"""Tests for the web server's OAuth layer.

Simulates what Claude.ai does — dynamic client registration, authorization
code + PKCE exchange, refresh — including a server restart in between,
which used to sign users out. Google is mocked; no network is needed.
Skipped when the 'web' extras are not installed.
"""

import base64
import hashlib
import os
import secrets
import time
import unittest
from datetime import datetime, timedelta
from unittest import mock

os.environ.setdefault("GOOGLE_CLIENT_ID", "test.apps.googleusercontent.com")
os.environ.setdefault("GOOGLE_CLIENT_SECRET", "test-google-client-secret")
os.environ.setdefault("BASE_URL", "https://ga-mcp.example.com")

try:
    from google.auth.exceptions import RefreshError
    from google.oauth2.credentials import Credentials
    from mcp.server.auth.provider import AuthorizationCode
    from starlette.testclient import TestClient

    from analytics_mcp import web_server
except ImportError as exc:  # pragma: no cover - web extras not installed
    web_server = None
    _IMPORT_ERROR = exc

REDIRECT = "https://claude.ai/api/mcp/auth_callback"
MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}
INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1"},
    },
}


def _fake_google_refresh(self, _request):
    self.token = "ya29.refreshed-" + secrets.token_hex(4)
    self.expiry = datetime.utcnow() + timedelta(hours=1)


@unittest.skipIf(web_server is None, "web extras not installed")
class TestWebServerAuth(unittest.TestCase):
    """Token lifecycle tests for the OAuth authorization server."""

    def setUp(self):
        os.environ.pop("SECRET_KEY", None)
        self.google_creds = {
            "token": "ya29.google-access",
            "refresh_token": "1//google-refresh",
            "token_uri": "https://oauth2.googleapis.com/token",
            "scopes": web_server._SCOPES,
            "expiry": int(time.time()) + 3600,
        }

    # --- helpers -----------------------------------------------------------

    def _register(self, client, auth_method="none"):
        r = client.post(
            "/register",
            json={
                "redirect_uris": [REDIRECT],
                "grant_types": ["authorization_code"],
                "token_endpoint_auth_method": auth_method,
            },
        )
        self.assertEqual(r.status_code, 201, r.text)
        return r.json()

    def _login(self, app, client, reg):
        """Mint an auth code the way /auth/callback does, then exchange it
        at /token with PKCE."""
        verifier = secrets.token_urlsafe(48)
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )
        provider = app.state.provider
        code = secrets.token_urlsafe(32)
        provider._auth_codes[code] = AuthorizationCode(
            code=code,
            scopes=web_server._SCOPES,
            expires_at=time.time() + 600,
            client_id=reg["client_id"],
            code_challenge=challenge,
            redirect_uri=REDIRECT,
            redirect_uri_provided_explicitly=True,
        )
        provider._auth_code_creds[code] = {
            "credentials": dict(self.google_creds),
            "email": "user@example.com",
        }
        form = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT,
            "client_id": reg["client_id"],
            "code_verifier": verifier,
        }
        if reg.get("client_secret"):
            form["client_secret"] = reg["client_secret"]
        r = client.post("/token", data=form)
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    @staticmethod
    def _refresh(client, reg, refresh_token):
        form = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": reg["client_id"],
        }
        if reg.get("client_secret"):
            form["client_secret"] = reg["client_secret"]
        return client.post("/token", data=form)

    @staticmethod
    def _mcp_status(client, access_token):
        r = client.post(
            "/mcp",
            json=INIT,
            headers={**MCP_HEADERS, "Authorization": f"Bearer {access_token}"},
        )
        return r.status_code

    # --- tests -------------------------------------------------------------

    def test_token_response_includes_refresh_token(self):
        app = web_server.create_app()
        with TestClient(app) as c:
            tokens = self._login(app, c, self._register(c))
            self.assertTrue(tokens.get("refresh_token"))
            self.assertEqual(self._mcp_status(c, tokens["access_token"]), 200)

    def test_survives_restart_and_refreshes(self):
        app1 = web_server.create_app()
        with TestClient(app1) as c1:
            reg = self._register(c1)
            tokens = self._login(app1, c1, reg)

        app2 = web_server.create_app()  # simulated redeploy
        with (
            TestClient(app2) as c2,
            mock.patch.object(Credentials, "refresh", _fake_google_refresh),
        ):
            self.assertEqual(self._mcp_status(c2, tokens["access_token"]), 200)
            r = self._refresh(c2, reg, tokens["refresh_token"])
            self.assertEqual(r.status_code, 200, r.text)
            new = r.json()
            self.assertEqual(self._mcp_status(c2, new["access_token"]), 200)
            payload = app2.state.provider.decode_access_token(
                new["access_token"]
            )
            self.assertTrue(
                payload["credentials"]["token"].startswith("ya29.refreshed-")
            )
            r2 = self._refresh(c2, reg, new["refresh_token"])
            self.assertEqual(r2.status_code, 200)

    def test_authorize_accepts_client_registered_before_restart(self):
        with TestClient(web_server.create_app()) as c1:
            reg = self._register(c1)
        with TestClient(web_server.create_app(), follow_redirects=False) as c2:
            r = c2.get(
                "/authorize",
                params={
                    "response_type": "code",
                    "client_id": reg["client_id"],
                    "redirect_uri": REDIRECT,
                    "code_challenge": "x" * 43,
                    "code_challenge_method": "S256",
                    "state": "s",
                },
            )
            self.assertEqual(r.status_code, 302, r.text)
            self.assertTrue(
                r.headers["location"].startswith("https://accounts.google.com/")
            )

    def test_confidential_client_survives_restart(self):
        app1 = web_server.create_app()
        with TestClient(app1) as c1:
            reg = self._register(c1, auth_method="client_secret_post")
            self.assertTrue(reg["client_secret"])
            tokens = self._login(app1, c1, reg)
        with (
            TestClient(web_server.create_app()) as c2,
            mock.patch.object(Credentials, "refresh", _fake_google_refresh),
        ):
            ok = self._refresh(c2, reg, tokens["refresh_token"])
            self.assertEqual(ok.status_code, 200)
            bad = {**reg, "client_secret": "wrong"}
            denied = self._refresh(c2, bad, tokens["refresh_token"])
            self.assertEqual(denied.status_code, 401)

    def test_revoked_google_grant_returns_invalid_grant(self):
        def revoked(_self, _req):
            raise RefreshError("invalid_grant: Token has been revoked.")

        app = web_server.create_app()
        with TestClient(app) as c:
            reg = self._register(c)
            tokens = self._login(app, c, reg)
            with mock.patch.object(Credentials, "refresh", revoked):
                r = self._refresh(c, reg, tokens["refresh_token"])
            self.assertEqual(r.status_code, 400)
            self.assertEqual(r.json()["error"], "invalid_grant")

    def test_transient_google_error_does_not_sign_out(self):
        def flaky(_self, _req):
            raise ConnectionError("network blip")

        app = web_server.create_app()
        with TestClient(app) as c, self.assertLogs(web_server.logger, "ERROR"):
            reg = self._register(c)
            tokens = self._login(app, c, reg)
            with mock.patch.object(Credentials, "refresh", flaky):
                r = self._refresh(c, reg, tokens["refresh_token"])
            self.assertEqual(r.status_code, 200, r.text)

    def test_token_types_are_not_interchangeable(self):
        app = web_server.create_app()
        with TestClient(app) as c:
            reg = self._register(c)
            tokens = self._login(app, c, reg)
            self.assertEqual(self._mcp_status(c, tokens["refresh_token"]), 401)
            r = self._refresh(c, reg, tokens["access_token"])
            self.assertEqual(r.status_code, 400)

    def test_refresh_token_bound_to_client(self):
        app = web_server.create_app()
        with TestClient(app) as c:
            tokens = self._login(app, c, self._register(c))
            other = self._register(c)
            r = self._refresh(c, other, tokens["refresh_token"])
            self.assertEqual(r.status_code, 400)

    def test_tampered_client_id_rejected(self):
        with TestClient(web_server.create_app()) as c:
            reg = self._register(c)
            r = c.post(
                "/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": "x",
                    "client_id": reg["client_id"][:-3] + "abc",
                },
            )
            self.assertEqual(r.status_code, 401)

    def test_unauthenticated_request_with_session_id_rejected(self):
        """Requests carrying an mcp-session-id used to skip auth."""
        with TestClient(web_server.create_app()) as c:
            r = c.post(
                "/mcp",
                json=INIT,
                headers={**MCP_HEADERS, "mcp-session-id": "anything"},
            )
            self.assertEqual(r.status_code, 401)

    def test_tokens_are_encrypted(self):
        app = web_server.create_app()
        with TestClient(app) as c:
            tokens = self._login(app, c, self._register(c))
        for token in (tokens["access_token"], tokens["refresh_token"]):
            raw = base64.urlsafe_b64decode(token + "==")
            self.assertNotIn(b"google-refresh", raw)

    def test_google_access_token_expiry_triggers_refresh(self):
        expired = web_server._creds_from_store(
            {**self.google_creds, "expiry": int(time.time()) - 60}
        )
        self.assertFalse(expired.valid)
        no_expiry = {
            k: v for k, v in self.google_creds.items() if k != "expiry"
        }
        self.assertFalse(web_server._creds_from_store(no_expiry).valid)
        self.assertTrue(web_server._creds_from_store(self.google_creds).valid)

    def test_secret_key_is_stable_without_env(self):
        self.assertEqual(
            web_server._resolve_secret_key(),
            web_server._resolve_secret_key(),
        )


if __name__ == "__main__":
    unittest.main()
