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



def _load_local_clip():
    """Load a free, local OpenCLIP model. First run downloads weights once."""
    try:
        import torch
        import open_clip
    except ImportError as exc:
        raise RuntimeError(
            "Local AI packages are missing. In the MoneyPrinterTurbo environment run: "
            "python -m pip install open_clip_torch"
        ) from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return torch, model, preprocess, tokenizer, device


def _local_ai_best_moment(sources, prompt, wanted_duration, work_folder, clip_bundle):
    """Rank candidate windows with local CLIP similarity and simple visual-quality checks."""
    from PIL import Image, ImageFilter, ImageStat

    torch, model, preprocess, tokenizer, device = clip_bundle
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("FFmpeg and FFprobe must be installed and available on PATH.")
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Add an AI visual prompt for this scene before analyzing it.")

    results = []
    for source_index, source in enumerate(sources, 1):
        probe = _run([
            ffprobe, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(source),
        ])
        try:
            source_duration = max(0.0, float(probe.stdout.strip()))
        except (ValueError, TypeError):
            source_duration = 0.0
        if source_duration <= 0:
            raise RuntimeError(f"Could not read duration for {source.name}")
        max_start = max(0.0, source_duration - min(wanted_duration, source_duration))
        step = 1.0
        sample_times = []
        t = 0.0
        while t <= source_duration + 0.001:
            sample_times.append(min(t, max(0.0, source_duration - 0.05)))
            t += step
        frame_dir = work_folder / f"ai_frames_{source_index:02d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        frames, qualities, timestamps = [], [], []
        for frame_index, timestamp in enumerate(sample_times):
            frame_path = frame_dir / f"frame_{frame_index:05d}.jpg"
            _run([
                ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                "-ss", f"{timestamp:.3f}", "-i", str(source),
                "-frames:v", "1", "-vf", "scale=320:-2", str(frame_path),
            ], timeout=120)
            if not frame_path.exists() or frame_path.stat().st_size == 0:
                continue
            with Image.open(frame_path) as image:
                rgb = image.convert("RGB")
                gray = rgb.convert("L")
                gray_stats = ImageStat.Stat(gray)
                brightness = gray_stats.mean[0] / 255.0
                # Edge energy is a lightweight blur/detail proxy; it is not semantic understanding.
                edge_energy = ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).stddev[0]
                detail = min(1.0, edge_energy / 48.0)
                exposure = max(0.0, 1.0 - abs(brightness - 0.5) / 0.5)
                quality = 0.65 * detail + 0.35 * exposure
                frames.append(preprocess(rgb))
                qualities.append(quality)
                timestamps.append(timestamp)
        if not frames:
            raise RuntimeError(f"No frames could be sampled from {source.name}")
        text_tokens = tokenizer([prompt]).to(device)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        similarities = []
        with torch.inference_mode():
            # Small batches keep VRAM usage reasonable on entry-level GPUs.
            for batch_start in range(0, len(frames), 16):
                image_tensor = torch.stack(frames[batch_start:batch_start + 16]).to(device)
                image_features = model.encode_image(image_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                similarities.extend(
                    (image_features @ text_features.T).squeeze(1).float().cpu().tolist()
                )

        low, high = min(similarities), max(similarities)
        spread = max(high - low, 1e-6)
        relevance = [(score - low) / spread for score in similarities]
        frame_scores = [0.8 * relevance[i] + 0.2 * qualities[i] for i in range(len(frames))]
        window_end = min(source_duration, wanted_duration)
        candidates = [0.0]
        candidate = step
        while candidate <= max_start + 0.001:
            candidates.append(candidate)
            candidate += step
        best = None
        for start in candidates:
            indices = [i for i, ts in enumerate(timestamps) if start <= ts < start + window_end]
            if not indices:
                indices = [min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - start))]
            score = sum(frame_scores[i] for i in indices) / len(indices)
            if best is None or score > best["score"]:
                best = {"start": start, "score": score, "indices": indices}
        best_frame_index = max(best["indices"], key=lambda i: frame_scores[i])
        results.append({
            "start": round(float(best["start"]), 3),
            "score": round(float(best["score"]), 4),
            "preview": str(frame_dir / f"frame_{timestamps.index(timestamps[best_frame_index]):05d}.jpg"),
            "source": str(source),
            "duration": round(source_duration, 3),
        })
    return results

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
        default_source_start = max(0.0, float(scene.get("source_start") or 0.0))
        ai_starts = scene.get("ai_source_starts") or []
        silent_clips = []

        # Each supplied source gets an equal share; AI-selected in-points override the default.
        for clip_index, source in enumerate(sources, 1):
            source_start = max(0.0, float(ai_starts[clip_index - 1])) if clip_index <= len(ai_starts) else default_source_start
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
            scene_voice = str(scene.get("voice") or "").strip()
            voice = str(settings.get("voice") or "ml-IN-MidhunNeural").strip() if scene_voice in ("", "ml-IN-MidhunNeural") else scene_voice
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


def _template_project(project_id="my_video"):
    return {
        "project_id": project_id,
        "settings": {
            "resolution": "480p",
            "aspect_ratio": "16:9",
            "fps": 24,
            "narration_speed": 0,
            "voice": "ml-IN-MidhunNeural",
            "crop_mode": "Fill and crop",
        },
        "scenes": [],
    }


def _validate_imported_project(data):
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object.")
    scenes = data.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError('JSON must contain a "scenes" array.')
    settings = data.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError('"settings" must be an object.')
    required = ("id", "start", "end")
    normalized = []
    for index, raw in enumerate(scenes, 1):
        if not isinstance(raw, dict):
            raise ValueError(f"Scene {index} must be a JSON object.")
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(f"Scene {index} is missing required field(s): {', '.join(missing)}.")
        try:
            start, end = float(raw["start"]), float(raw["end"])
        except (TypeError, ValueError):
            raise ValueError(f"Scene {index} start/end must be numbers.")
        if start < 0 or end <= start:
            raise ValueError(f"Scene {index} must have start >= 0 and end > start.")
        urls = raw.get("urls", raw.get("url", ""))
        if isinstance(urls, list):
            urls = "\n".join(str(url) for url in urls)
        elif not isinstance(urls, str):
            raise ValueError(f"Scene {index} urls must be a string or array of URLs.")
        normalized.append({
            "id": str(raw.get("id") or f"scene_{index}"),
            "start": start,
            "end": end,
            "urls": urls,
            "local_path": str(raw.get("local_path") or ""),
            "keywords": str(raw.get("keywords") or ""),
            "visual_prompt": str(raw.get("visual_prompt") or "Describe the exact subject and visual style to match."),
            "source_start": float(raw.get("source_start") or 0),
            "narration": str(raw.get("narration") or ""),
            "voice": str(raw.get("voice") or ""),
        })
    return {
        "project_id": str(data.get("project_id") or "my_video"),
        "settings": settings,
        "scenes": normalized,
    }


def main():
    st.set_page_config(page_title="Manual Timeline Editor", page_icon="🎬", layout="wide")
    st.title("🎬 Manual Timeline Editor")
    st.caption("Build a scene-by-scene timeline with accordions, Malayalam narration, video sources, local AI analysis, and JSON import/export.")
    st.info(
        "Open a scene accordion to edit all of its options. Paste one HTTPS direct video URL or Pexels video-page URL per line. "
        "The editor uses every listed clip in order and divides that scene's duration equally across them."
    )

    if "manual_scene_rows" not in st.session_state:
        st.session_state.manual_scene_rows = []
    if "manual_project_settings" not in st.session_state:
        st.session_state.manual_project_settings = _template_project()["settings"].copy()
    if "manual_project_id" not in st.session_state:
        st.session_state.manual_project_id = "my_video"
    if "manual_ai_results" not in st.session_state:
        st.session_state.manual_ai_results = {}
    if "manual_ai_starts" not in st.session_state:
        st.session_state.manual_ai_starts = {}

    widget_version = int(st.session_state.get("manual_widget_version", 0))
    with st.sidebar:
        st.session_state.manual_project_id = st.text_input(
            "Project name", st.session_state.manual_project_id, key=f"manual_project_name_input_{widget_version}"
        )
        st.caption("Scene data can be saved to or restored from JSON below.")

    settings = st.session_state.manual_project_settings
    st.subheader("Project settings")
    with st.expander("Video resolution, aspect ratio, frame rate and narration defaults", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            settings["resolution"] = st.selectbox(
                "Output resolution", ["480p", "720p", "1080p", "1440p", "2160p (4K)"],
                index=["480p", "720p", "1080p", "1440p", "2160p (4K)"].index(settings.get("resolution", "480p"))
                if settings.get("resolution", "480p") in ["480p", "720p", "1080p", "1440p", "2160p (4K)"] else 0,
                key=f"manual_setting_resolution_{widget_version}",
            )
            settings["aspect_ratio"] = st.selectbox(
                "Aspect ratio", ["16:9", "9:16", "1:1"],
                index=["16:9", "9:16", "1:1"].index(settings.get("aspect_ratio", "16:9"))
                if settings.get("aspect_ratio", "16:9") in ["16:9", "9:16", "1:1"] else 0,
                key=f"manual_setting_aspect_{widget_version}",
            )
        with c2:
            settings["fps"] = st.selectbox(
                "Frame rate (FPS)", [24, 25, 30, 50, 60],
                index=[24, 25, 30, 50, 60].index(settings.get("fps", 24))
                if settings.get("fps", 24) in [24, 25, 30, 50, 60] else 0,
                key=f"manual_setting_fps_{widget_version}",
            )
            settings["narration_speed"] = st.slider(
                "Narration speed (%)", -50, 100, int(settings.get("narration_speed", 0)), 5,
                key=f"manual_setting_narration_speed_{widget_version}",
            )
        with c3:
            settings["voice"] = st.selectbox(
                "Default TTS voice",
                ["ml-IN-MidhunNeural", "ml-IN-SobhanaNeural", "en-US-AriaNeural", "en-US-GuyNeural", "en-GB-SoniaNeural"],
                index=["ml-IN-MidhunNeural", "ml-IN-SobhanaNeural", "en-US-AriaNeural", "en-US-GuyNeural", "en-GB-SoniaNeural"].index(settings.get("voice", "ml-IN-MidhunNeural"))
                if settings.get("voice", "ml-IN-MidhunNeural") in ["ml-IN-MidhunNeural", "ml-IN-SobhanaNeural", "en-US-AriaNeural", "en-US-GuyNeural", "en-GB-SoniaNeural"] else 0,
                key=f"manual_setting_voice_{widget_version}",
            )
            settings["crop_mode"] = st.selectbox(
                "Video fitting", ["Fill and crop", "Fit with bars"],
                index=["Fill and crop", "Fit with bars"].index(settings.get("crop_mode", "Fill and crop"))
                if settings.get("crop_mode", "Fill and crop") in ["Fill and crop", "Fit with bars"] else 0,
                key=f"manual_setting_crop_mode_{widget_version}",
            )
    st.session_state.manual_project_settings = settings

    st.subheader("Scene JSON — import, template and export")
    template = _template_project(st.session_state.manual_project_id or "my_video")
    current_project = {
        "project_id": st.session_state.manual_project_id or "my_video",
        "settings": settings,
        "scenes": [
            {key: row.get(key, default) for key, default in {
                "id": "", "start": 0, "end": 8, "urls": "", "local_path": "",
                "keywords": "", "visual_prompt": "", "source_start": 0,
                "narration": "", "voice": "",
            }.items()}
            for row in st.session_state.manual_scene_rows
        ],
    }
    j1, j2, j3 = st.columns(3)
    with j1:
        st.download_button(
            "Download JSON template" if not st.session_state.manual_scene_rows else "Download current scenes JSON",
            data=json.dumps(current_project if st.session_state.manual_scene_rows else template, ensure_ascii=False, indent=2),
            file_name="manual_timeline_template.json" if not st.session_state.manual_scene_rows else "manual_timeline_project.json",
            mime="application/json",
            use_container_width=True,
        )
    with j2:
        uploaded_json = st.file_uploader("Upload scene JSON", type=["json"], key="manual_scene_json_upload")
    with j3:
        if uploaded_json is not None and st.button("Load JSON and create scenes", use_container_width=True, type="primary"):
            try:
                imported = _validate_imported_project(json.loads(uploaded_json.getvalue().decode("utf-8-sig")))
                st.session_state.manual_project_id = imported["project_id"]
                st.session_state.manual_project_settings = {
                    **_template_project()["settings"], **imported["settings"]
                }
                st.session_state.manual_scene_rows = imported["scenes"]
                st.session_state.manual_ai_results = {}
                st.session_state.manual_ai_starts = {}
                # Give widgets new keys so imported values override old browser widget state.
                st.session_state.manual_widget_version = widget_version + 1
                st.success(f"Loaded {len(imported['scenes'])} scene(s) from JSON.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not load scene JSON: {exc}")

    st.caption(
        'JSON format: {"project_id":"birds_test","settings":{...},"scenes":[{"id":"scene_1","start":0,"end":15,"urls":["https://..."],"visual_prompt":"...","narration":"...","voice":"ml-IN-MidhunNeural"}]}. '
        "The importer also accepts urls as a newline-separated string."
    )

    st.subheader("Scenes")
    if st.button("➕ Add scene", use_container_width=True):
        rows = st.session_state.manual_scene_rows
        last_end = max([float(row.get("end", 0) or 0) for row in rows], default=0.0)
        inherited_voice = (rows[0].get("voice") or settings.get("voice") or "ml-IN-MidhunNeural") if rows else (settings.get("voice") or "ml-IN-MidhunNeural")
        rows.append({
            "id": f"scene_{len(rows) + 1}", "start": last_end, "end": last_end + 15,
            "urls": "", "local_path": "", "keywords": "",
            "visual_prompt": "Describe the exact subject and visual style to match.",
            "source_start": 0.0, "narration": "", "voice": inherited_voice,
        })
        st.session_state.manual_scene_rows = rows
        st.rerun()

    voices = [
        "ml-IN-MidhunNeural", "ml-IN-SobhanaNeural",
        "en-US-AriaNeural", "en-US-GuyNeural", "en-GB-SoniaNeural",
    ]
    for index, row in enumerate(st.session_state.manual_scene_rows):
        sid = re.sub(r"[^A-Za-z0-9_-]", "_", str(row.get("id") or f"scene_{index + 1}"))
        row.setdefault("voice", (st.session_state.manual_scene_rows[0].get("voice") if st.session_state.manual_scene_rows else "") or settings.get("voice") or "ml-IN-MidhunNeural")
        row.setdefault("urls", "")
        row.setdefault("local_path", "")
        row.setdefault("keywords", "")
        row.setdefault("visual_prompt", "Describe the exact subject and visual style to match.")
        row.setdefault("source_start", 0.0)
        row.setdefault("narration", "")
        with st.expander(f"🎬 {row.get('id') or f'Scene {index + 1}'} · {row.get('start', 0)}s–{row.get('end', 0)}s", expanded=(index == 0)):
            a, b, c = st.columns(3)
            with a:
                row["id"] = st.text_input("Scene ID", str(row.get("id") or f"scene_{index + 1}"), key=f"scene_id_{widget_version}_{index}")
                row["start"] = st.number_input("Timeline start (seconds)", min_value=0.0, value=float(row.get("start", 0) or 0), step=1.0, key=f"scene_start_{widget_version}_{index}")
            with b:
                row["end"] = st.number_input("Timeline end (seconds)", min_value=0.1, value=max(0.1, float(row.get("end", 15) or 15)), step=1.0, key=f"scene_end_{widget_version}_{index}")
                row["source_start"] = st.number_input("Default source start (seconds)", min_value=0.0, value=float(row.get("source_start", 0) or 0), step=1.0, key=f"scene_source_start_{widget_version}_{index}")
            with c:
                current_voice = row.get("voice") or (st.session_state.manual_scene_rows[0].get("voice") if st.session_state.manual_scene_rows else "") or settings.get("voice") or "ml-IN-MidhunNeural"
                if current_voice not in voices:
                    voices = voices + [current_voice]
                row["voice"] = st.selectbox(
                    "Narration voice", voices,
                    index=voices.index(current_voice),
                    key=f"scene_voice_{widget_version}_{index}",
                    help="New scenes inherit Scene 1's selected voice by default. You can override any scene individually.",
                )
            row["narration"] = st.text_area("Narration text", str(row.get("narration") or ""), height=130, key=f"scene_narration_{widget_version}_{index}")
            row["urls"] = st.text_area(
                "Video URLs (one per line, in playback order)", str(row.get("urls") or ""),
                height=140, key=f"scene_urls_{widget_version}_{index}",
                placeholder="https://www.pexels.com/video/example-video-12345/\nhttps://cdn.example.com/clip.mp4",
            )
            row["local_path"] = st.text_input("Optional local video file path", str(row.get("local_path") or ""), key=f"scene_local_path_{widget_version}_{index}")
            row["keywords"] = st.text_input("Optional Pexels fallback search keywords", str(row.get("keywords") or ""), key=f"scene_keywords_{widget_version}_{index}")
            row["visual_prompt"] = st.text_area(
                "AI visual prompt", str(row.get("visual_prompt") or ""), height=80, key=f"scene_prompt_{widget_version}_{index}",
                help="Describe the exact subject, activity, and visual style that the free local AI should match.",
            )
            st.caption("All URLs are used in the listed order. If URLs are supplied, local path and keywords are ignored.")
            if st.button("Remove this scene", key=f"remove_scene_{index}"):
                removed_id = sid
                st.session_state.manual_scene_rows.pop(index)
                st.session_state.manual_ai_results.pop(removed_id, None)
                st.session_state.manual_ai_starts.pop(removed_id, None)
                st.rerun()

    pexels_ready = bool(_configured_pexels_key())
    if pexels_ready:
        st.success("Pexels API key found in MoneyPrinterTurbo config (or environment).")
    else:
        st.warning("No Pexels key found. Add one in the main MoneyPrinterTurbo settings/config, or use direct HTTPS video URLs/local files.")
    st.subheader("Free local AI — best-moment analysis")
    st.caption("Uses local OpenCLIP to compare sampled frames against each scene prompt. First run may download model weights.")
    st.warning("This ranks windows within each source; it does not automatically discard mismatched URLs. Review previews before rendering.")

    if st.button("Analyze best moments (free local AI)", use_container_width=True, disabled=not st.session_state.manual_scene_rows):
        try:
            clip_bundle = _load_local_clip()
            progress = st.progress(0.0, text="Loading local AI model…")
            rows = st.session_state.manual_scene_rows
            for scene_index, row in enumerate(rows, 1):
                sid = re.sub(r"[^A-Za-z0-9_-]", "_", str(row.get("id") or f"scene_{scene_index}"))
                scene_folder = OUT / (st.session_state.manual_project_id or "my_video") / f"scene_{scene_index:03d}_{sid}"
                sources = _resolve_sources(row, scene_folder, sid)
                scene_duration = float(row["end"]) - float(row["start"])
                if scene_duration <= 0:
                    raise ValueError(f"{sid}: end time must be greater than start time.")
                per_clip = scene_duration / max(1, len(sources))
                prompt = str(row.get("visual_prompt") or row.get("keywords") or row.get("narration") or "").strip()
                results = _local_ai_best_moment(sources, prompt, per_clip, scene_folder, clip_bundle)
                st.session_state.manual_ai_starts[sid] = [item["start"] for item in results]
                st.session_state.manual_ai_results[sid] = results
                progress.progress(scene_index / max(1, len(rows)), text=f"Analyzed {sid}")
            st.success("Analysis complete. Review previews, then render.")
        except Exception as exc:
            st.error(f"AI analysis failed: {exc}")

    for index, row in enumerate(st.session_state.manual_scene_rows):
        sid = re.sub(r"[^A-Za-z0-9_-]", "_", str(row.get("id") or f"scene_{index + 1}"))
        results = st.session_state.manual_ai_results.get(sid, [])
        if results:
            with st.expander(f"AI preview — {row.get('id', f'scene_{index + 1}')}", expanded=True):
                for clip_index, item in enumerate(results, 1):
                    st.markdown(f"**Clip {clip_index}:** start at {item['start']:.1f}s · score {item['score']:.3f} · duration {item['duration']:.1f}s")
                    if Path(item["preview"]).is_file():
                        st.image(item["preview"], caption=f"Selected frame from clip {clip_index}", width=360)
                if st.button("Clear AI selection for this scene", key=f"clear_ai_{index}"):
                    st.session_state.manual_ai_starts.pop(sid, None)
                    st.session_state.manual_ai_results.pop(sid, None)
                    st.rerun()

    project_id = re.sub(r"[^A-Za-z0-9_-]", "_", st.session_state.manual_project_id or "my_video") or "my_video"
    project_scenes = []
    for index, row in enumerate(st.session_state.manual_scene_rows, 1):
        scene = dict(row)
        sid = re.sub(r"[^A-Za-z0-9_-]", "_", str(scene.get("id") or f"scene_{index}"))
        scene["ai_source_starts"] = st.session_state.manual_ai_starts.get(sid, [])
        project_scenes.append(scene)
    project = {"project_id": project_id, "settings": settings, "scenes": project_scenes}

    left, right = st.columns(2)
    with left:
        if st.button("Save scene plan JSON to disk", use_container_width=True):
            path = OUT / project_id / "scene_plan.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
            st.success(f"Saved: {path}")
    with right:
        if st.button("Render video", type="primary", use_container_width=True, disabled=not project_scenes):
            try:
                output = render(project, OUT / project_id)
                st.video(str(output))
                st.download_button("Download rendered video", output.read_bytes(), file_name=output.name, mime="video/mp4")
            except Exception as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
