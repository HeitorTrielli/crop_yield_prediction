"""
Google Drive API helpers: auth, list folder, download file, delete file.

Used by download_soy_gee_drive.py for GEE export downloads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

SCOPES_DRIVE = ["https://www.googleapis.com/auth/drive"]


def build_drive_service(credentials_dir: Path) -> Any:
    """Build Drive API service using credentials_dir/credentials.json and token.json."""
    credentials_dir = credentials_dir.resolve()
    creds_file = credentials_dir / "credentials.json"
    token_file = credentials_dir / "token.json"

    if not creds_file.exists():
        raise FileNotFoundError(
            f"Drive credentials not found: {creds_file}. "
            "Download OAuth client JSON from Google Cloud Console and save as credentials.json"
        )

    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES_DRIVE)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(creds_file), SCOPES_DRIVE
            )
            creds = flow.run_local_server(port=0)
        credentials_dir.mkdir(parents=True, exist_ok=True)
        with open(token_file, "w") as f:
            f.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


def get_folder_id_by_name(service: Any, folder_name: str) -> Optional[str]:
    """Return Drive folder ID for folder in root, or None if not found."""
    try:
        q = (
            f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' "
            "and 'root' in parents and trashed = false"
        )
        resp = service.files().list(q=q, pageSize=1, fields="files(id, name)").execute()
        items = resp.get("files", [])
        return items[0]["id"] if items else None
    except HttpError as e:
        raise RuntimeError(f"Drive API error listing folder: {e}") from e


def list_files_in_folder(service: Any, folder_id: str) -> list[dict]:
    """List all files in a Drive folder (not trashed)."""
    out = []
    page_token = None
    while True:
        resp = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                pageSize=100,
                fields="nextPageToken, files(id, name)",
                pageToken=page_token,
            )
            .execute()
        )
        out.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return out


def download_file_from_drive(
    service: Any, file_id: str, file_name: str, local_path: Path
) -> None:
    """Download one file from Drive to local_path/file_name."""
    local_path = Path(local_path)
    local_path.mkdir(parents=True, exist_ok=True)
    dest = local_path / file_name
    request = service.files().get_media(fileId=file_id)
    with open(dest, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def delete_file_from_drive(service: Any, file_id: str) -> None:
    """Permanently delete a file from Drive."""
    service.files().delete(fileId=file_id).execute()
