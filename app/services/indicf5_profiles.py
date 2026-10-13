"""Persistent profiles for the local IndicF5 HTTP service."""
import json
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = ROOT / "storage" / "indicf5_profiles"
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")
DEFAULT_API_URL = os.environ.get("INDICF5_API_URL", "http://127.0.0.1:8765").rstrip("/")
DEFAULT_REFERENCE_DIR = os.environ.get(
    "INDICF5_REFERENCE_DIR", r"C:\AI\IndicF5\reference"
)


def _make_id(value: str) -> str:
    value = re.sub(r"[^a-z0-9_-]+", "-", str(value or "").strip().lower()).strip("-_")
    value = re.sub(r"[-_]{2,}", "-", value)
    if not value or not _ID_RE.fullmatch(value):
        raise ValueError("Use letters, numbers, spaces, hyphens or underscores for the profile name.")
    return value


def list_profiles() -> list[dict]:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    profiles = []
    for path in sorted(PROFILE_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            profile_id = str(data.get("id") or path.stem)
            if not _ID_RE.fullmatch(profile_id):
                continue
            profiles.append({
                "id": profile_id,
                "name": str(data.get("name") or profile_id),
                "reference_audio": str(data.get("reference_audio") or ""),
                "reference_text": str(data.get("reference_text") or ""),
                "api_url": str(data.get("api_url") or DEFAULT_API_URL).rstrip("/"),
            })
        except (OSError, ValueError, TypeError):
            continue
    return profiles


def get_profile(profile_id: str) -> dict:
    profile_id = str(profile_id or "").strip()
    if not _ID_RE.fullmatch(profile_id):
        raise ValueError("Invalid IndicF5 voice profile ID.")
    path = PROFILE_DIR / f"{profile_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"IndicF5 profile '{profile_id}' was not found.")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["id"] = profile_id
    data["api_url"] = str(data.get("api_url") or DEFAULT_API_URL).rstrip("/")
    return data


def save_profile(name: str, reference_audio: str, reference_text: str,
                 api_url: str = DEFAULT_API_URL, profile_id: str | None = None) -> str:
    display_name = str(name or "").strip()
    audio_path = Path(str(reference_audio or "").strip()).expanduser()
    transcript = str(reference_text or "").strip()
    api_url = str(api_url or DEFAULT_API_URL).strip().rstrip("/")
    if not display_name or len(display_name) > 80:
        raise ValueError("Enter a profile name between 1 and 80 characters.")
    if not audio_path.is_file():
        raise ValueError(f"Reference WAV was not found: {audio_path}")
    if audio_path.suffix.lower() != ".wav":
        raise ValueError("IndicF5 reference audio must be a WAV file.")
    if not transcript:
        raise ValueError("Enter the exact transcript spoken in the reference audio.")
    parsed = urlparse(api_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Enter a valid IndicF5 API base URL, for example http://127.0.0.1:8765.")
    profile_id = _make_id(profile_id or display_name)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "id": profile_id,
        "name": display_name,
        "reference_audio": str(audio_path.resolve()),
        "reference_text": transcript,
        "api_url": api_url,
    }
    target = PROFILE_DIR / f"{profile_id}.json"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=PROFILE_DIR,
                                     suffix=".tmp", delete=False) as handle:
        temp_path = Path(handle.name)
        json.dump(data, handle, ensure_ascii=False, indent=2)
    try:
        temp_path.replace(target)
    finally:
        temp_path.unlink(missing_ok=True)
    return profile_id


def delete_profile(profile_id: str) -> None:
    profile_id = str(profile_id or "").strip()
    if not _ID_RE.fullmatch(profile_id):
        raise ValueError("Invalid IndicF5 voice profile ID.")
    (PROFILE_DIR / f"{profile_id}.json").unlink(missing_ok=True)
