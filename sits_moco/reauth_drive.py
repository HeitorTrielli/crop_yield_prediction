"""Force a fresh Google Drive OAuth login and verify API access."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO / "data_download"))

from drive_api import build_drive_service

CREDS_DIR = _REPO / "data"
TOKEN = CREDS_DIR / "token.json"
BR_DWGD_FOLDER = "11-qnvwojirAtaQxSE03N0_SUrbcsz44N"


def main() -> None:
    if not (CREDS_DIR / "credentials.json").is_file():
        raise SystemExit(
            f"Missing {CREDS_DIR / 'credentials.json'}.\n"
            "Create an OAuth Desktop client in Google Cloud Console, enable Drive API,\n"
            "download the JSON, and save it as data/credentials.json."
        )
    if TOKEN.is_file():
        TOKEN.unlink()
        print(f"Removed stale {TOKEN}")
    print("Starting Drive login (browser should open)…")
    service = build_drive_service(CREDS_DIR)
    about = service.about().get(fields="user").execute()
    user = about.get("user", {})
    print("OK — authenticated as", user.get("emailAddress") or user.get("displayName"))
    resp = (
        service.files()
        .list(
            q=f"'{BR_DWGD_FOLDER}' in parents and trashed = false",
            pageSize=15,
            fields="files(id, name, size)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        .execute()
    )
    files = resp.get("files", [])
    print(f"BR-DWGD folder reachable ({len(files)} files shown):")
    for item in files:
        print(f"  {item.get('name')}  ({item.get('size', '?')} bytes)")


if __name__ == "__main__":
    main()
