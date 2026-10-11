"""Opt-in manual timeline editor for scene-by-scene video assembly."""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "storage" / "manual_timeline"




def _get_footage(scene, folder, sid):
    """Resolve explicit Pexels/direct MP4 URLs, local files, then keyword search."""
    import requests
    from urllib.parse import urlparse
    url = str(scene.get("url") or "").strip()
    local = str(scene.get("local_path") or "").strip()
    keywords = str(scene.get("keywords") or "").strip()
    key = os.environ.get("PEXELS_API_KEY", "").strip()

    if url:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        direct = url
        if parsed.scheme != "https":
            raise ValueError(f"{sid}: only HTTPS video URLs are supported.")
        if host.endswith("pexels.com") and "/video/" in parsed.path:
            match = re.search(r"-(\d+)/?$", parsed.path)
            if not match:
                raise ValueError(f"{sid}: could not identify the Pexels video ID.")
            if not key:
                raise ValueError("Set PEXELS_API_KEY to resolve Pexels page links.")
            response = requests.get("https://api.pexels.com/videos/videos/" + match.group(1),
                                    headers={"Authorization": key}, timeout=30)
            response.raise_for_status()
            files = [v for v in response.json().get("video_files", [])
                     if v.get("link") and v.get("file_type") == "video/mp4"]
            if not files:
                raise ValueError(f"{sid}: Pexels returned no MP4.")
            files.sort(key=lambda v: abs(int(v.get("width") or 0) - 1920))
            direct = files[0]["link"]
        elif not parsed.path.lower().endswith((".mp4", ".m4v", ".mov", ".webm")):
            raise ValueError(f"{sid}: use a Pexels video page or a direct video-file URL.")
        target = folder / "media" / (sid + ".mp4")
        target.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(direct, stream=True, timeout=60) as response:
            response.raise_for_status()
            with target.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        output.write(chunk)
        if not target.exists() or target.stat().st_size == 0:
            raise ValueError(f"{sid}: video download was empty.")
        return target

    if local:
        source = Path(local).expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"{sid}: local video file does not exist: {local}")
        return source

    if not keywords:
        raise ValueError(f"{sid}: provide a video URL, local path, or fallback keywords.")
    if not key:
        raise ValueError("Set PEXELS_API_KEY to enable keyword search.")
    response = requests.get("https://api.pexels.com/videos/search",
        params={"query": keywords, "per_page": 10, "orientation": "landscape"},
        headers={"Authorization": key}, timeout=30)
    response.raise_for_status()
    videos = response.json().get("videos", [])
    if not videos:
        raise ValueError(f"{sid}: no Pexels results for keyword: {keywords}")
    files = [v for v in videos[0].get("video_files", [])
             if v.get("link") and v.get("file_type") == "video/mp4"]
    if not files:
        raise ValueError(f"{sid}: Pexels result had no MP4.")
    files.sort(key=lambda v: abs(int(v.get("width") or 0) - 1920))
    return _get_footage({"url": files[0]["link"]}, folder, sid)


def render(project, folder):
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("FFmpeg and FFprobe must be installed and available on PATH.")
    scenes = sorted(project["scenes"], key=lambda x: float(x["start"]))
    if not scenes:
        raise ValueError("Add at least one scene.")
    cursor = 0.0
    for scene in scenes:
        start, end = float(scene["start"]), float(scene["end"])
        if start < cursor - .001:
            raise ValueError("Scene time slots overlap.")
        if start > cursor + .001:
            raise ValueError("Timeline has a gap; make slots contiguous starting at 0.")
        if end <= start:
            raise ValueError("Each scene end must be after its start.")
        cursor = end
    folder.mkdir(parents=True, exist_ok=True)
    parts = []
    for index, scene in enumerate(scenes, 1):
        sid = re.sub(r"[^A-Za-z0-9_-]", "_", str(scene.get("id") or f"scene_{index}"))
        source = _get_footage(scene, folder, sid)
        start, end = float(scene["start"]), float(scene["end"])
        duration = end - start
        part = folder / f"{index:03d}_{sid}.mp4"
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-ss", str(float(scene.get("source_start") or 0)), "-stream_loop", "-1", "-i", str(source)]
        narration = str(scene.get("narration") or "").strip()
        if narration:
            try:
                import edge_tts
            except ImportError as exc:
                raise RuntimeError("Install edge-tts to generate narration: pip install edge-tts") from exc
            import asyncio
            audio = folder / f"{index:03d}_{sid}.mp3"
            asyncio.run(edge_tts.Communicate(narration, str(scene.get("voice") or "ml-IN-MidhunNeural")).save(str(audio)))
            cmd += ["-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-af", f"apad,atrim=duration={duration}"]
        else:
            cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-map", "0:v:0", "-map", "1:a:0"]
        cmd += ["-vf", "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,fps=30",
                "-t", str(duration), "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-ar", "48000", "-ac", "2", str(part)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode:
            raise RuntimeError(result.stderr[-2000:])
        parts.append(part)
    concat = folder / "concat.txt"
    concat.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in parts), encoding="utf-8")
    output = folder / (re.sub(r"[^A-Za-z0-9_-]", "_", project["project_id"]) + ".mp4")
    result = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(output)], capture_output=True, text=True, timeout=1800)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])
    return output


def main():
    st.set_page_config(page_title="Manual Timeline Editor", page_icon="🎬", layout="wide")
    st.title("🎬 Manual Timeline Editor")
    st.caption("Separate from the existing automatic generation mode. Set exact scene times and bind narration to each scene.")
    st.info("Source priority: explicit URL → local file path → keyword search. If an explicit URL fails, rendering stops instead of silently replacing it with another clip.")
    with st.sidebar:
        project_id = st.text_input("Project name", "my_video")
    if "manual_scene_rows" not in st.session_state:
        st.session_state.manual_scene_rows = [{"id":"scene_1","start":0.0,"end":8.0,"url":"","local_path":"","keywords":"","source_start":0.0,"narration":"","voice":"ml-IN-MidhunNeural"}]
    rows = st.data_editor(st.session_state.manual_scene_rows, num_rows="dynamic", hide_index=True, use_container_width=True,
        column_config={
            "id":st.column_config.TextColumn("Scene ID", required=True),
            "start":st.column_config.NumberColumn("Timeline start (s)", min_value=0.0, step=1.0),
            "end":st.column_config.NumberColumn("Timeline end (s)", min_value=0.1, step=1.0),
            "url":st.column_config.TextColumn("Video URL (Pexels page/direct MP4)", width="large"),
            "local_path":st.column_config.TextColumn("Local video file path", width="large"),
            "keywords":st.column_config.TextColumn("Fallback keywords", width="medium"),
            "source_start":st.column_config.NumberColumn("Source in-point (s)", min_value=0.0, step=1.0),
            "narration":st.column_config.TextColumn("Narration (Malayalam supported)", width="large"),
            "voice":st.column_config.TextColumn("TTS voice"),
        })
    st.session_state.manual_scene_rows = rows.to_dict("records") if hasattr(rows, "to_dict") else list(rows)
    st.caption("Pexels page links and keyword search require PEXELS_API_KEY. Direct HTTPS MP4 links and local files do not need the key.")
    st.code("set PEXELS_API_KEY=your_key_here", language="bat")
    project_id = re.sub(r"[^A-Za-z0-9_-]", "_", project_id) or "my_video"
    project = {"project_id":project_id, "scenes":st.session_state.manual_scene_rows}
    a, b = st.columns(2)
    with a:
        if st.button("Save scene plan", use_container_width=True):
            path = OUT / project_id / "scene_plan.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
            st.success(f"Saved: {path}")
    with b:
        if st.button("Render video", type="primary", use_container_width=True):
            try:
                output = render(project, OUT / project_id)
                st.video(str(output))
                st.download_button("Download rendered video", output.read_bytes(), file_name=output.name, mime="video/mp4")
            except Exception as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
