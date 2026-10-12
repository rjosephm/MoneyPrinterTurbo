"""Reference Voice Studio: save and test reusable VoxCPM voice profiles."""
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import config
from app.services import voice, voice_profiles
from app.utils import utils


st.title("Reference Voice Studio")
st.caption("Save a reusable speaker reference and test Malayalam narration before using it in a video.")
st.info(
    "Upload a voice recording you own or have permission to use. This creates a reusable reference profile; "
    "it does not train a new model. Voice synthesis uses your configured VoxCPM API endpoint."
)

api_ready = bool(
    str(config.voxcpm.get("api_key", "") or "").strip()
    and str(config.voxcpm.get("model_id", "") or "").strip()
)
if api_ready:
    st.success("VoxCPM API key and model ID are configured.")
else:
    st.warning("Configure the VoxCPM API key and model ID in MoneyPrinterTurbo Settings before generating a test. You can still save a profile now.")

profiles = voice_profiles.list_profiles()
if profiles:
    st.subheader("Saved voice profiles")
    for profile in profiles:
        with st.expander(profile["name"] + "  ·  " + profile["id"]):
            try:
                st.audio(voice_profiles.load_profile_audio(profile["id"]), format="audio/wav")
            except (OSError, ValueError) as exc:
                st.error(str(exc))
            if st.button("Delete profile", key="delete_voice_profile_" + profile["id"]):
                voice_profiles.delete_profile(profile["id"])
                st.success("Profile deleted.")
                st.rerun()

st.divider()
st.subheader("Create or test a voice profile")
profile_choices = ["__new__"] + [item["id"] for item in profiles]
selected_existing_profile = st.selectbox(
    "Start from an existing profile (optional)",
    profile_choices,
    format_func=lambda profile_id: (
        "Create a new profile from an upload"
        if profile_id == "__new__"
        else next((item["name"] for item in profiles if item["id"] == profile_id), profile_id)
    ),
    key="voice_studio_existing_profile",
)
existing_profile = next(
    (item for item in profiles if item["id"] == selected_existing_profile), None
)
profile_name = st.text_input(
    "Profile name",
    value=existing_profile["name"] if existing_profile else "My Malayalam Voice",
    max_chars=80,
    key="voice_studio_profile_name_" + selected_existing_profile,
)
uploaded = st.file_uploader(
    "Upload or replace reference recording",
    type=list(voice.VOXCPM_REFERENCE_AUDIO_FILE_TYPES),
    help="Use a clean recording with one speaker and minimal background music/noise. Maximum 20 MiB and 120 seconds.",
    key="voice_studio_reference_upload",
)

reference_wav = None
if uploaded is not None:
    raw = uploaded.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    cached = st.session_state.get("voice_studio_normalized")
    if isinstance(cached, dict) and cached.get("digest") == digest:
        reference_wav = cached.get("wav")
    else:
        try:
            with st.spinner("Preparing reference audio…"):
                reference_wav = voice.prepare_voxcpm_reference_audio(raw, Path(uploaded.name).suffix)
            st.session_state["voice_studio_normalized"] = {"digest": digest, "wav": reference_wav}
        except Exception as exc:
            st.session_state.pop("voice_studio_normalized", None)
            st.error("Could not prepare the recording: " + str(exc))
elif existing_profile:
    try:
        reference_wav = voice_profiles.load_profile_audio(existing_profile["id"])
    except (OSError, ValueError) as exc:
        st.error(str(exc))

if reference_wav:
    st.audio(reference_wav, format="audio/wav")
    st.caption("Reference audio: normalized mono, 16 kHz WAV. The original upload is not modified.")

test_text = st.text_area(
    "Test narration",
    value="നമസ്കാരം സുഹൃത്തുക്കളേ, ഇന്ന് നമുക്ക് അതിശയകരമായ അഞ്ച് മനോഹരമായ പക്ഷികളെ പരിചയപ്പെടാം.",
    height=110,
)
pitch_col, speed_col = st.columns(2)
with pitch_col:
    pitch_semitones = st.slider("Pitch adjustment (semitones)", -4, 4, 0, 1)
with speed_col:
    speed_percent = st.slider("Speaking speed adjustment (%)", -20, 20, 0, 5)
st.caption("Pitch and speed adjustments apply to the test preview only. The saved profile preserves the reference speaker identity.")
save_col, test_col = st.columns(2)
with save_col:
    if st.button("Save voice profile", type="primary", use_container_width=True, disabled=not bool(reference_wav)):
        try:
            profile_id = voice_profiles.save_profile(profile_name, reference_wav)
            st.session_state["voice_studio_selected_profile"] = profile_id
            st.success(f"Saved profile: {profile_name} ({profile_id})")
            st.rerun()
        except (ValueError, OSError) as exc:
            st.error(str(exc))
with test_col:
    if st.button("Generate test narration", use_container_width=True, disabled=not (reference_wav and api_ready and bool(test_text.strip()))):
        try:
            with tempfile.TemporaryDirectory(prefix="mpt-voice-studio-") as temp_dir:
                raw_audio_path = Path(temp_dir) / "voice_test_raw.mp3"
                audio_path = Path(temp_dir) / "voice_test.mp3"
                with st.spinner("Generating a test with VoxCPM…"):
                    result = voice.tts(
                        text=test_text.strip(),
                        voice_name="voxcpm:default",
                        voice_rate=1.0,
                        voice_file=str(raw_audio_path),
                        voice_volume=1.0,
                        voxcpm_reference_audio=reference_wav,
                    )
                if result is None or not raw_audio_path.is_file() or raw_audio_path.stat().st_size == 0:
                    st.error("VoxCPM did not return audio. Check the API key, model ID, endpoint, and MoneyPrinterTurbo logs.")
                else:
                    if pitch_semitones or speed_percent:
                        ffmpeg = utils.get_ffmpeg_binary() or shutil.which("ffmpeg")
                        if not ffmpeg:
                            raise RuntimeError("FFmpeg is required for pitch/speed preview adjustments.")
                        pitch_ratio = 2 ** (pitch_semitones / 12.0)
                        speed_ratio = 1.0 + speed_percent / 100.0
                        filters = (
                            f"asetrate=48000*{pitch_ratio:.8f},aresample=48000,"
                            f"atempo={1.0 / pitch_ratio:.8f},atempo={speed_ratio:.3f}"
                        )
                        processed = subprocess.run(
                            [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                             "-i", str(raw_audio_path), "-af", filters, "-c:a", "libmp3lame",
                             "-q:a", "2", str(audio_path)],
                            capture_output=True, text=True, timeout=180,
                        )
                        if processed.returncode or not audio_path.is_file() or audio_path.stat().st_size == 0:
                            raise RuntimeError((processed.stderr or "FFmpeg audio processing failed")[-2000:])
                    else:
                        audio_path = raw_audio_path
                    audio_bytes = audio_path.read_bytes()
                    st.session_state["voice_studio_test_audio"] = audio_bytes
        except Exception as exc:
            st.error("Voice test failed: " + str(exc))

preview_audio = st.session_state.get("voice_studio_test_audio")
if preview_audio:
    st.subheader("Latest test narration")
    st.audio(preview_audio, format="audio/mp3")
    st.download_button(
        "Download latest test narration",
        preview_audio,
        file_name="voice_studio_test.mp3",
        mime="audio/mpeg",
        key="download_latest_voice_studio_test",
    )

st.caption("Saved profiles are stored locally under storage/voice_profiles. Manual Timeline and automatic generation can use the same profile. Reference audio is sent to the configured VoxCPM endpoint when you generate speech.")
