"""Manage saved profiles for the user's local IndicF5 service."""
from pathlib import Path
import sys

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services import indicf5_profiles

st.title("IndicF5 Voice Profiles")
st.caption("Save reference audio and its exact transcript once, then select the saved profile in Automatic Voiceover or Manual Timeline.")

st.info(
    "The local IndicF5 API must be running at http://127.0.0.1:8765. "
    "Reference audio and transcript are saved as a reusable profile; you won't need to enter them for every video."
)

profiles = indicf5_profiles.list_profiles()
if profiles:
    st.subheader("Saved profiles")
    for profile in profiles:
        with st.expander(f"{profile['name']}  ·  {profile['id']}", expanded=False):
            st.write("Reference WAV:", profile["reference_audio"])
            st.write("Reference transcript:", profile["reference_text"])
            st.write("API:", profile["api_url"])
            if st.button("Edit this profile", key=f"edit_indicf5_{profile['id']}"):
                st.session_state["indicf5_edit_profile_id"] = profile["id"]
                st.rerun()
            if st.button("Delete profile", key=f"delete_indicf5_{profile['id']}"):
                indicf5_profiles.delete_profile(profile["id"])
                st.session_state.pop("indicf5_edit_profile_id", None)
                st.success("Profile deleted.")
                st.rerun()
else:
    st.warning("No saved IndicF5 profiles yet. Your existing malayalam_ref.wav can be used to create the first one.")

edit_id = st.session_state.get("indicf5_edit_profile_id")
editing = next((p for p in profiles if p["id"] == edit_id), None)
st.divider()
st.subheader("Edit saved profile" if editing else "Create a profile")
with st.form("indicf5_profile_form"):
    name = st.text_input("Profile name", value=editing["name"] if editing else "My IndicF5 Malayalam")
    audio_path = st.text_input(
        "Reference WAV path",
        value=editing["reference_audio"] if editing else r"C:\AI\IndicF5\reference\malayalam_ref.wav",
        help="Use a WAV file accessible to the IndicF5 service. Your current normalized reference WAV is the default.",
    )
    reference_text = st.text_area(
        "Exact transcript of the reference audio",
        value=editing["reference_text"] if editing else "ഇനി ഗ്രോത് ഹോർമോണിന് എന്തെങ്കിലും പ്രബലമായ ഒരു പ്രഭാവം ബ്രോയ്ലർ കോഴികളുടെ വളർച്ചയിൽ ഉണ്ടാക്കണം",
        height=100,
    )
    api_url = st.text_input("IndicF5 API base URL", value=editing["api_url"] if editing else indicf5_profiles.DEFAULT_API_URL)
    save = st.form_submit_button("Save profile", type="primary")
if save:
    try:
        profile_id = indicf5_profiles.save_profile(
            name, audio_path, reference_text, api_url,
            profile_id=editing["id"] if editing else None,
        )
        st.session_state.pop("indicf5_edit_profile_id", None)
        st.success(f"Saved profile: {name} ({profile_id})")
        st.rerun()
    except (OSError, ValueError) as exc:
        st.error(str(exc))

st.divider()
st.subheader("Check local API")
health_url = st.text_input("Health-check URL", value=indicf5_profiles.DEFAULT_API_URL + "/health")
if st.button("Check API"):
    try:
        response = requests.get(health_url, timeout=5)
        response.raise_for_status()
        st.json(response.json())
    except Exception as exc:
        st.error(f"Cannot reach IndicF5 API: {exc}")
