"""Persistent, reusable reference-audio voice profiles for MoneyPrinterTurbo."""
import json
import re
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = ROOT / "storage" / "voice_profiles"
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")


def _profile_id(value: str) -> str:
    value = str(value or "").strip().lower()
    value = re.sub(r"[^a-z0-9_-]+", "-", value).strip("-_")
    value = re.sub(r"[-_]{2,}", "-", value)
    if not value or not _ID_RE.fullmatch(value):
        raise ValueError("Use a profile name containing letters, numbers, spaces, hyphens or underscores.")
    return value


def list_profiles() -> list[dict]:
    """Return profile metadata only; audio bytes are loaded on demand."""
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    result = []
    for metadata_path in sorted(PROFILE_DIR.glob("*.json")):
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            profile_id = str(data.get("id") or metadata_path.stem)
            if not _ID_RE.fullmatch(profile_id):
                continue
            audio_path = PROFILE_DIR / f"{profile_id}.wav"
            if audio_path.is_file() and audio_path.stat().st_size:
                result.append({"id": profile_id, "name": str(data.get("name") or profile_id)})
        except (OSError, ValueError, TypeError):
            continue
    return result


def load_profile_audio(profile_id: str) -> bytes:
    """Load a previously normalized 16 kHz mono WAV profile."""
    profile_id = str(profile_id or "").strip()
    if not _ID_RE.fullmatch(profile_id):
        raise ValueError("Invalid voice profile ID.")
    path = PROFILE_DIR / f"{profile_id}.wav"
    if not path.is_file():
        raise FileNotFoundError(f"Voice profile '{profile_id}' was not found.")
    data = path.read_bytes()
    if not data or not data.startswith(b"RIFF") or data[8:12] != b"WAVE":
        raise ValueError(f"Voice profile '{profile_id}' is not a valid WAV file.")
    return data


def save_profile(name: str, wav_audio: bytes) -> str:
    """Save an already-normalized reference WAV and metadata atomically."""
    display_name = str(name or "").strip()
    if not display_name or len(display_name) > 80:
        raise ValueError("Enter a voice profile name (1–80 characters).")
    if not isinstance(wav_audio, bytes) or not wav_audio.startswith(b"RIFF") or wav_audio[8:12] != b"WAVE":
        raise ValueError("The reference audio must be a normalized WAV file.")
    profile_id = _profile_id(display_name)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {"id": profile_id, "name": display_name, "format": "wav", "sample_rate": 16000, "channels": 1}
    with tempfile.NamedTemporaryFile(dir=PROFILE_DIR, suffix=".tmp", delete=False) as handle:
        audio_tmp = Path(handle.name)
        handle.write(wav_audio)
    metadata_tmp = PROFILE_DIR / f".{profile_id}.json.tmp"
    try:
        metadata_tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        audio_tmp.replace(PROFILE_DIR / f"{profile_id}.wav")
        metadata_tmp.replace(PROFILE_DIR / f"{profile_id}.json")
    finally:
        audio_tmp.unlink(missing_ok=True)
        metadata_tmp.unlink(missing_ok=True)
    return profile_id


def delete_profile(profile_id: str) -> None:
    """Delete only the named profile's own metadata and WAV files."""
    profile_id = str(profile_id or "").strip()
    if not _ID_RE.fullmatch(profile_id):
        raise ValueError("Invalid voice profile ID.")
    (PROFILE_DIR / f"{profile_id}.wav").unlink(missing_ok=True)
    (PROFILE_DIR / f"{profile_id}.json").unlink(missing_ok=True)
