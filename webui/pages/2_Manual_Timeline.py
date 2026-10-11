"""Opt-in manual timeline editor with multiple clips and editable per-scene narration."""
import asyncio
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

from app.config import config

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "storage" / "manual_timeline"


def _configured_pexels_key():
    """Reuse the API keys already configured in MoneyPrinterTurbo."""
    configured = config.app.get("pexels_api_keys", [])
    if isinstance(configured, str):
        configured = [item.strip() for item in configured.split(",") if item.strip()]
    if isinstance(configured, (list, tuple)):
        for item in configured:
            if str(item).strip():
                return str(item).strip()
    return os.environ.get("PEXELS_API_KEY", "").strip()


def _output_size(settings):
    resolution = settings.get("resolution", "1080p")
    aspect = settings.get("aspect_ratio", "16:9")
    widths = {"480p": 854, "720p": 1280, "1080p": 1920, "1440p": 2560, "2160p (4K)": 3840}
    heights = {"480p": 480, "720p": 720, "1080p": 1080, "1440p": 1440, "2160p (4K)": 2160}
    height = heights.get(resolution, 1080)
    width = widths.get(resolution, 1920)
    if aspect == "9:16":
        return height, width
    if aspect == "1:1":
        return height, height
    return width, height


def _split_urls(value):
    """Accept one URL per line, or URLs separated by semicolons."""
    raw = str(value or "").replace("\r", "\n")
    parts = []
    for line in raw.split("\n"):
        parts.extend(line.split(";"))
    return [part.strip() for part in parts if part.strip()]


def _download_video_url(url, target, sid):
    import requests

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https":
        raise ValueError(f"{sid}: only HTTPS video URLs are supported.")

    direct = url
    if host.endswith("pexels.com") and "/video/" in parsed.path:
        match = re.search(r"-(\d+)/?$", parsed.path)
        if not match:
            raise ValueError(f"{sid}: could not identify the Pexels video ID in {url}")
        key = _configured_pexels_key()
        if not key:
            raise ValueError("Set PEXELS_API_KEY to resolve Pexels page links.")
        response = requests.get(
            "https://api.pexels.com/videos/videos/" + match.group(1),
            headers={"Authorization": key}, timeout=30,
        )
        response.raise_for_status()
        files = [
            item for item in response.json().get("video_files", [])
            if item.get("link") and item.get("file_type") == "video/mp4"
        ]
        if not files:
            raise ValueError(f"{sid}: Pexels returned no MP4 for {url}")
        files.sort(key=lambda item: abs(int(item.get("width") or 0) - 1920))
        direct = files[0]["link"]
    elif not parsed.path.lower().endswith((".mp4", ".m4v", ".mov", ".webm")):
        raise ValueError(f"{sid}: use a Pexels video page or a direct video-file URL: {url}")

    target.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(direct, stream=True, timeout=60) as response:
        response.raise_for_status()
        with target.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    output.write(chunk)
    if not target.exists() or target.stat().st_size == 0:
        raise ValueError(f"{sid}: download was empty for {url}")
    return target


def _resolve_sources(scene, folder, sid):
    urls = _split_urls(scene.get("urls") or scene.get("url"))
    local = str(scene.get("local_path") or "").strip()
    keywords = str(scene.get("keywords") or "").strip()

    # Explicit URLs take priority. Every supplied URL must work; never silently substitute.
    if urls:
        sources = []
        for index, url in enumerate(urls, 1):
            sources.append(_download_video_url(
                url, folder / "media" / f"{sid}_source_{index:02d}.mp4", sid
            ))
        return sources

    if local:
        source = Path(local).expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"{sid}: local video file does not exist: {local}")
        return [source]

    if not keywords:
        raise ValueError(f"{sid}: provide video URLs, a local path, or fallback keywords.")

    key = _configured_pexels_key()
    if not key:
        raise ValueError("No Pexels API key found in MoneyPrinterTurbo config or PEXELS_API_KEY environment variable.")
    import requests
    response = requests.get(
        "https://api.pexels.com/videos/search",
        params={"query": keywords, "per_page": 10, "orientation": "landscape"},
        headers={"Authorization": key}, timeout=30,
    )
    response.raise_for_status()
    videos = response.json().get("videos", [])
    if not videos:
        raise ValueError(f"{sid}: no Pexels results for keyword: {keywords}")
    files = [
        item for item in videos[0].get("video_files", [])
        if item.get("link") and item.get("file_type") == "video/mp4"
    ]
    if not files:
        raise ValueError(f"{sid}: Pexels result had no MP4.")
    files.sort(key=lambda item: abs(int(item.get("width") or 0) - 1920))
    return [_download_video_url(
        files[0]["link"], folder / "media" / f"{sid}_keyword_result.mp4", sid
    )]


def _run(cmd, timeout=1800):
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-3000:])
    return result


def render(project, folder):
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("FFmpeg and FFprobe must be installed and available on PATH.")

    scenes = sorted(project["scenes"], key=lambda row: float(row["start"]))
    if not scenes:
        raise ValueError("Add at least one scene.")
    cursor = 0.0
    for scene in scenes:
        start, end = float(scene["start"]), float(scene["end"])
        if start < cursor - 0.001:
            raise ValueError("Scene time slots overlap.")
        if start > cursor + 0.001:
            raise ValueError("Timeline has a gap; make slots contiguous starting at 0.")
        if end <= start:
            raise ValueError("Each scene end must be after its start.")
        cursor = end

    settings = project.get("settings", {})
    width, height = _output_size(settings)
    fps = int(settings.get("fps", 30))
    narration_rate = int(settings.get("narration_speed", 0))
    tts_rate = f"{narration_rate:+d}%"
    if settings.get("crop_mode", "Fill and crop") == "Fit with bars":
        video_filter = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={fps},setsar=1"
    else:
        video_filter = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},fps={fps},setsar=1"
    folder.mkdir(parents=True, exist_ok=True)
    scene_parts = []
    for scene_index, scene in enumerate(scenes, 1):
        sid = re.sub(r"[^A-Za-z0-9_-]", "_", str(scene.get("id") or f"scene_{scene_index}"))
        scene_folder = folder / f"scene_{scene_index:03d}_{sid}"
        scene_folder.mkdir(parents=True, exist_ok=True)
        sources = _resolve_sources(scene, scene_folder, sid)
        start, end = float(scene["start"]), float(scene["end"])
        duration = end - start
        per_clip = duration / len(sources)
        source_start = max(0.0, float(scene.get("source_start") or 0.0))
        silent_clips = []

        # Each supplied source gets an equal share of the scene; short sources loop to fill it.
        for clip_index, source in enumerate(sources, 1):
            clip_path = scene_folder / f"clip_{clip_index:02d}.mp4"
            _run([
                ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                "-stream_loop", "-1", "-ss", str(source_start), "-i", str(source),
                "-map", "0:v:0", "-an",
                "-vf", video_filter,
                "-t", f"{per_clip:.6f}", "-c:v", "libx264", "-preset", "veryfast",
                "-pix_fmt", "yuv420p", "-video_track_timescale", "90000",
                str(clip_path),
            ])
            silent_clips.append(clip_path)

        concat_list = scene_folder / "clips.txt"
        concat_list.write_text(
            "".join(f"file '{path.resolve().as_posix()}'\n" for path in silent_clips),
            encoding="utf-8",
        )
        stitched = scene_folder / "stitched_silent.mp4"
        _run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-t", f"{duration:.6f}", "-c", "copy", str(stitched),
        ])

        narration = str(scene.get("narration") or "").strip()
        audio_path = scene_folder / "narration.mp3"
        if narration:
            try:
                import edge_tts
            except ImportError as exc:
                raise RuntimeError("Install edge-tts to generate narration: pip install edge-tts") from exc
            voice = str(scene.get("voice") or settings.get("voice") or "ml-IN-MidhunNeural").strip()
            asyncio.run(edge_tts.Communicate(narration, voice, rate=tts_rate).save(str(audio_path)))
            audio_input = str(audio_path)
        else:
            audio_input = "anullsrc=r=48000:cl=stereo"

        scene_output = folder / f"{scene_index:03d}_{sid}.mp4"
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", str(stitched),
        ]
        if narration:
            cmd += ["-i", audio_input]
        else:
            cmd += ["-f", "lavfi", "-i", audio_input]
        _run(cmd + [
            "-map", "0:v:0", "-map", "1:a:0",
            "-af", f"apad,atrim=duration={duration:.6f}",
            "-t", f"{duration:.6f}", "-c:v", "copy", "-c:a", "aac",
            "-ar", "48000", "-ac", "2", "-movflags", "+faststart", str(scene_output),
        ])
        scene_parts.append(scene_output)

    concat = folder / "timeline.txt"
    concat.write_text(
        "".join(f"file '{path.resolve().as_posix()}'\n" for path in scene_parts),
        encoding="utf-8",
    )
    output = folder / (re.sub(r"[^A-Za-z0-9_-]", "_", project["project_id"]) + ".mp4")
    _run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat),
        "-c", "copy", "-movflags", "+faststart", str(output),
    ])
    return output


def main():
    st.set_page_config(page_title="Manual Timeline Editor", page_icon="🎬", layout="wide")
    st.title("🎬 Manual Timeline Editor")
    st.caption("Set exact scene times, paste multiple clip URLs per scene, and edit each scene's narration before rendering.")
    st.info(
        "Paste one video URL per line in a scene's URL box. The editor splits that scene's duration evenly "
        "across the supplied clips, center-crops to 16:9, and stitches them in the order listed. "
        "Explicit URLs are never silently replaced if one fails."
    )
    with st.sidebar:
        project_id = st.text_input("Project name", "my_video")

    st.subheader("Video Settings")
    with st.expander("Resolution, aspect ratio, frame rate and narration", expanded=True):
        setting1, setting2, setting3 = st.columns(3)
        with setting1:
            resolution = st.selectbox("Output resolution", ["480p", "720p", "1080p", "1440p", "2160p (4K)"], index=2)
            aspect_ratio = st.selectbox("Aspect ratio", ["16:9", "9:16", "1:1"], index=0)
        with setting2:
            fps = st.selectbox("Frame rate (FPS)", [24, 25, 30, 50, 60], index=2)
            narration_speed = st.slider("Narration speed (%)", min_value=-50, max_value=100, value=0, step=5, help="0% is normal speed; negative is slower and positive is faster.")
        with setting3:
            tts_voice = st.text_input("TTS voice", "ml-IN-MidhunNeural", help="Use an Edge TTS voice ID, for example ml-IN-MidhunNeural.")
            crop_mode = st.selectbox("Video fitting", ["Fill and crop", "Fit with bars"], index=0)

    settings = {
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
        "fps": int(fps),
        "narration_speed": int(narration_speed),
        "voice": tts_voice.strip() or "ml-IN-MidhunNeural",
        "crop_mode": crop_mode,
    }

    if "manual_scene_rows" not in st.session_state:
        st.session_state.manual_scene_rows = [{
            "id": "scene_1", "start": 0.0, "end": 8.0, "urls": "",
            "local_path": "", "keywords": "", "source_start": 0.0,
            "narration": "", "voice": "ml-IN-MidhunNeural",
        }]

    edited = st.data_editor(
        st.session_state.manual_scene_rows,
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_order=["id", "start", "end", "local_path", "keywords", "source_start", "narration", "voice"],
        column_config={
            "id": st.column_config.TextColumn("Scene ID", required=True),
            "start": st.column_config.NumberColumn("Timeline start (s)", min_value=0.0, step=1.0),
            "end": st.column_config.NumberColumn("Timeline end (s)", min_value=0.1, step=1.0),
            "urls": None,
            "local_path": st.column_config.TextColumn("Local video file path", width="large"),
            "keywords": st.column_config.TextColumn("Fallback keywords", width="medium"),
            "source_start": st.column_config.NumberColumn("Source in-point (s)", min_value=0.0, step=1.0),
            "narration": st.column_config.TextColumn("Editable narration (Malayalam supported)", width="large"),
            "voice": st.column_config.TextColumn("TTS voice"),
        },
    )
    st.session_state.manual_scene_rows = edited.to_dict("records") if hasattr(edited, "to_dict") else list(edited)

    st.subheader("Video URLs per scene")
    st.caption("Paste one HTTPS direct video URL or Pexels video-page URL per line. Pexels page links and keyword search require PEXELS_API_KEY.")
    for index, row in enumerate(st.session_state.manual_scene_rows):
        scene_id = str(row.get("id") or f"scene_{index + 1}")
        row.setdefault("urls", row.get("url", ""))
        with st.expander(f"{scene_id} — {row.get('start', 0)} to {row.get('end', 0)} sec", expanded=(index == 0)):
            row["urls"] = st.text_area(
                "Video URLs (one per line; clips are stitched in this order)",
                value=str(row.get("urls") or ""),
                key=f"manual_urls_{index}_{scene_id}",
                height=120,
                placeholder="https://example.com/clip1.mp4\nhttps://example.com/clip2.mp4",
            )

    pexels_ready = bool(_configured_pexels_key())
    if pexels_ready:
        st.success("Pexels API key found in MoneyPrinterTurbo config (or environment).")
    else:
        st.warning("No Pexels key found. Add one in the main MoneyPrinterTurbo settings/config, or use direct HTTPS video URLs/local files.")
    st.caption("The editor reuses app.pexels_api_keys from MoneyPrinterTurbo's config; it does not require you to paste the key into this page.")
    project_id = re.sub(r"[^A-Za-z0-9_-]", "_", project_id) or "my_video"
    project = {"project_id": project_id, "settings": settings, "scenes": st.session_state.manual_scene_rows}

    left, right = st.columns(2)
    with left:
        if st.button("Save scene plan", use_container_width=True):
            path = OUT / project_id / "scene_plan.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
            st.success(f"Saved: {path}")
    with right:
        if st.button("Render video", type="primary", use_container_width=True):
            try:
                output = render(project, OUT / project_id)
                st.video(str(output))
                st.download_button("Download rendered video", output.read_bytes(), file_name=output.name, mime="video/mp4")
            except Exception as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
