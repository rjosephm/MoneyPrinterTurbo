"""Reference Voice Studio: save and test reusable VoxCPM voice profiles."""
import hashlib
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import config
from app.services import voice, voice_profiles


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
st.subheader("Create or update a voice profile")
profile_name = st.text_input("Profile name", value="My Malayalam Voice", max_chars=80)
uploaded = st.file_uploader(
    "Upload reference recording",
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
        except (ValueError, OSError, RuntimeError) as exc:
            st.session_state.pop("voice_studio_normalized", None)
            st.error("Could not prepare the recording: " + str(exc))
    if reference_wav:
        st.audio(reference_wav, format="audio/wav")
        st.caption("Normalized reference: mono, 16 kHz WAV. The original upload is not modified.")

test_text = st.text_area(
    "Test narration",
    value="നമസ്കാരം സുഹൃത്തുക്കളേ, ഇന്ന് നമുക്ക് അതിശയകരമായ അഞ്ച് മനോഹരമായ പക്ഷികളെ പരിചയപ്പെടാം.",
    height=110,
)
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
                audio_path = Path(temp_dir) / "voice_test.mp3"
                with st.spinner("Generating a test with VoxCPM…"):
                    result = voice.tts(
                        text=test_text.strip(),
                        voice_name="voxcpm:default",
                        voice_rate=1.0,
                        voice_file=str(audio_path),
                        voice_volume=1.0,
                        voxcpm_reference_audio=reference_wav,
                    )
                if result is None or not audio_path.is_file() or audio_path.stat().st_size == 0:
                    st.error("VoxCPM did not return audio. Check the API key, model ID, endpoint, and MoneyPrinterTurbo logs.")
                else:
                    audio_bytes = audio_path.read_bytes()
                    st.session_state["voice_studio_test_audio"] = audio_bytes
                    st.audio(audio_bytes, format="audio/mp3")
                    st.download_button("Download test narration", audio_bytes, file_name="voice_studio_test.mp3", mime="audio/mpeg")
        except Exception as exc:
            st.error("Voice test failed: " + str(exc))

st.caption("Saved profiles are stored locally under storage/voice_profiles. Manual Timeline and automatic generation can use the same profile. Reference audio is sent to the configured VoxCPM endpoint when you generate speech.")
