from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path

from config import Settings
from core.exceptions import UploadError


class YouTubeAuth:
    """Owns YouTube OAuth token loading, refresh, creation, and service building."""

    SCOPES = [
        "https://www.googleapis.com/auth/youtube.upload",
        "https://www.googleapis.com/auth/youtube.force-ssl",
    ]

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger

    def get_authenticated_service(self):
        """Return an authenticated YouTube Data API v3 service object."""
        try:
            from googleapiclient.discovery import build
        except ImportError as exc:
            self.logger.error("Authentication failed: Google API dependencies are missing")
            raise UploadError("YouTube dependencies are missing. Run: pip install -r requirements.txt") from exc

        credentials = self.get_credentials()
        self.logger.info("YouTube authentication successful")
        return build("youtube", "v3", credentials=credentials)

    def get_credentials(self):
        """Load, refresh, or create OAuth credentials."""
        try:
            self._hydrate_client_secret_from_environment()
            self._validate_client_secret()
            self._hydrate_token_from_environment()
            credentials = self._load_existing_token()

            if credentials and credentials.expired and credentials.refresh_token:
                credentials = self._refresh_token(credentials)

            if credentials and credentials.valid:
                self.logger.info("Reusing existing YouTube token: %s", self.settings.youtube_token_file)
                return credentials

            if self._is_headless_environment():
                raise UploadError(
                    "YouTube token is missing or invalid in a headless environment. "
                    "Run OAuth locally once, then deploy storage/credentials/youtube_token.json "
                    "or provide it as YOUTUBE_TOKEN_JSON/YOUTUBE_TOKEN_BASE64."
                )

            return self._create_token_interactively()
        except UploadError:
            self.logger.error("YouTube authentication failed")
            raise
        except Exception as exc:
            self.logger.error("YouTube authentication failed: %s", exc)
            raise UploadError(f"YouTube authentication failed: {exc}") from exc

    def validate_authentication(self) -> None:
        """Fail fast if upload authentication cannot be completed."""
        self.get_credentials()

    def _validate_client_secret(self) -> None:
        path = self.settings.youtube_client_secrets_file
        if not path:
            raise UploadError("YOUTUBE_CLIENT_SECRETS_FILE is required.")
        if not path.exists():
            raise UploadError(
                f"YouTube OAuth client secret file is missing: {path}. "
                "Place your OAuth client JSON at storage/credentials/client_secret.json "
                "or set YOUTUBE_CLIENT_SECRETS_FILE."
            )
        self.logger.info("Detected YouTube OAuth client secrets file: %s", path)

    def _hydrate_client_secret_from_environment(self) -> None:
        path = self.settings.youtube_client_secrets_file
        if path and path.exists() and path.stat().st_size > 0:
            return

        client_payload = self.settings.youtube_client_secrets_json
        if self.settings.youtube_client_secrets_base64:
            client_payload = base64.b64decode(self.settings.youtube_client_secrets_base64).decode("utf-8")

        if not client_payload:
            return

        try:
            parsed = json.loads(client_payload)
        except json.JSONDecodeError as exc:
            raise UploadError("YOUTUBE_CLIENT_SECRETS_JSON/YOUTUBE_CLIENT_SECRETS_BASE64 is not valid JSON.") from exc

        if not path:
            raise UploadError("YOUTUBE_CLIENT_SECRETS_FILE is required when using deployment client-secret env.")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
        self.logger.info("YouTube OAuth client secrets created from deployment secret at: %s", path)

    def _hydrate_token_from_environment(self) -> None:
        token_file = self.settings.youtube_token_file
        if token_file.exists() and token_file.stat().st_size > 0:
            return

        token_payload = self.settings.youtube_token_json
        if self.settings.youtube_token_base64:
            token_payload = base64.b64decode(self.settings.youtube_token_base64).decode("utf-8")

        if not token_payload:
            return

        try:
            parsed = json.loads(token_payload)
        except json.JSONDecodeError as exc:
            raise UploadError("YOUTUBE_TOKEN_JSON/YOUTUBE_TOKEN_BASE64 is not valid JSON.") from exc

        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
        self.logger.info("YouTube token created from deployment secret at: %s", token_file)

    def _load_existing_token(self):
        token_file = self.settings.youtube_token_file
        if not token_file.exists() or token_file.stat().st_size == 0:
            return None

        try:
            token_payload = json.loads(token_file.read_text(encoding="utf-8"))
            stored_scopes = set(token_payload.get("scopes") or token_payload.get("scope") or [])
            if isinstance(token_payload.get("scope"), str):
                stored_scopes = set(token_payload["scope"].split())
            missing_scopes = set(self.SCOPES) - stored_scopes
            if missing_scopes:
                if self._is_headless_environment():
                    raise UploadError(
                        "Existing YouTube token is missing required scopes. "
                        "Re-authenticate locally with python main.py --auth-only and redeploy youtube_token.json."
                    )
                self.logger.warning(
                    "Existing YouTube token is missing required scopes; local OAuth will recreate it"
                )
                return None

            from google.oauth2.credentials import Credentials

            credentials = Credentials.from_authorized_user_file(str(token_file), self.SCOPES)
            return credentials
        except Exception as exc:
            if self._is_headless_environment():
                raise UploadError(f"Existing YouTube token is unreadable in headless mode: {exc}") from exc
            self.logger.warning("Existing YouTube token is unreadable; local OAuth will recreate it: %s", exc)
            return None

    def _refresh_token(self, credentials):
        try:
            from google.auth.transport.requests import Request

            credentials.refresh(Request())
            self._save_token(credentials)
            self.logger.info("YouTube token refreshed")
            return credentials
        except Exception as exc:
            if self._is_headless_environment():
                raise UploadError(
                    "YouTube token refresh failed in headless mode. "
                    "Generate a fresh youtube_token.json locally and redeploy it."
                ) from exc
            self.logger.warning("YouTube token refresh failed; local OAuth will recreate it: %s", exc)
            return None

    def _create_token_interactively(self):
        try:
            from google_auth_oauthlib.flow import InstalledAppFlow

            flow = InstalledAppFlow.from_client_secrets_file(
                str(self.settings.youtube_client_secrets_file),
                self.SCOPES,
            )
            self.logger.info("Opening browser for first-time YouTube OAuth consent")
            credentials = flow.run_local_server(
                port=0,
                open_browser=True,
                access_type="offline",
                prompt="consent",
                authorization_prompt_message="Open this URL if the browser does not open automatically: {url}",
                success_message="YouTube authentication completed. You can close this window.",
            )
            self._save_token(credentials)
            self.logger.info("YouTube token created: %s", self.settings.youtube_token_file)
            return credentials
        except Exception as exc:
            raise UploadError(f"Could not complete local YouTube OAuth flow: {exc}") from exc

    def _save_token(self, credentials) -> None:
        token_file = self.settings.youtube_token_file
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(credentials.to_json(), encoding="utf-8")

    def _is_headless_environment(self) -> bool:
        explicit = os.getenv("HEADLESS", "").strip().lower()
        if explicit in {"1", "true", "yes", "on"}:
            return True
        if explicit in {"0", "false", "no", "off"}:
            return False
        cloud_markers = (
            "CI",
            "RAILWAY_ENVIRONMENT",
            "RAILWAY_PROJECT_ID",
            "RENDER",
            "RENDER_SERVICE_ID",
            "K_SERVICE",
            "DYNO",
        )
        if any(os.getenv(marker) for marker in cloud_markers):
            return True
        return os.name != "nt" and not os.getenv("DISPLAY")
