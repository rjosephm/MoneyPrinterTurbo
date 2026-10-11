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
        source = Path(str(scene.get("local_path") or "")).expanduser()
        if not str(scene.get("local_path") or "").strip() or not source.is_file():
            raise ValueError(f"Scene {sid}: enter an existing local video path. URL downloads are added in the next stage.")
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
    st.info("Current first stage supports local video paths. URL resolution/download and keyword fallback are the next implementation stage; this page will never substitute a different clip silently.")
    with st.sidebar:
        project_id = st.text_input("Project name", "my_video")
    if "manual_scene_rows" not in st.session_state:
        st.session_state.manual_scene_rows = [{"id":"scene_1","start":0.0,"end":8.0,"local_path":"","source_start":0.0,"narration":"","voice":"ml-IN-MidhunNeural"}]
    rows = st.data_editor(st.session_state.manual_scene_rows, num_rows="dynamic", hide_index=True, use_container_width=True,
        column_config={
            "id":st.column_config.TextColumn("Scene ID", required=True),
            "start":st.column_config.NumberColumn("Timeline start (s)", min_value=0.0, step=1.0),
            "end":st.column_config.NumberColumn("Timeline end (s)", min_value=0.1, step=1.0),
            "local_path":st.column_config.TextColumn("Local video file path", width="large"),
            "source_start":st.column_config.NumberColumn("Source in-point (s)", min_value=0.0, step=1.0),
            "narration":st.column_config.TextColumn("Narration (Malayalam supported)", width="large"),
            "voice":st.column_config.TextColumn("TTS voice"),
        })
    st.session_state.manual_scene_rows = rows.to_dict("records") if hasattr(rows, "to_dict") else list(rows)
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
