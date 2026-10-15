"""Local FFmpeg post-processing for generated narration; no cloud services required."""
import shutil
import subprocess
from pathlib import Path


def _run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-3000:])
    return result


def probe_duration(path, ffprobe="ffprobe"):
    result = _run([
        ffprobe, "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ])
    try:
        return max(0.0, float(result.stdout.strip()))
    except (ValueError, TypeError):
        return 0.0


def build_filter(settings):
    """Build a conservative, CPU-only filter chain for speech."""
    settings = settings or {}
    filters = []
    clarity = max(0, min(100, int(settings.get("clarity", 45))))
    if clarity:
        filters.extend([
            "highpass=f=75",
            "equalizer=f=250:t=q:w=1:g=-1.0",
            f"equalizer=f=3000:t=q:w=1:g={clarity * 0.035:.2f}",
        ])
    pitch = max(-2.0, min(2.0, float(settings.get("pitch_semitones", 0.0))))
    if abs(pitch) > 0.001:
        ratio = 2 ** (pitch / 12.0)
        filters.extend([
            f"asetrate={int(round(48000 * ratio))}",
            "aresample=48000",
            f"atempo={1.0 / ratio:.8f}",
        ])
    speed = max(0.5, min(2.0, float(settings.get("speed", 1.0))))
    if abs(speed - 1.0) > 0.001:
        filters.append(f"atempo={speed:.4f}")
    target = max(-23, min(-14, int(settings.get("target_lufs", -16))))
    filters.append(f"loudnorm=I={target}:TP=-1.0:LRA=11")
    filters.append("aresample=48000")
    return ",".join(filters)


def process_audio(source, destination, settings=None, ffmpeg="ffmpeg"):
    """Write processed PCM WAV to a distinct destination; never overwrite source."""
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if source == destination:
        raise ValueError("Processed audio destination must differ from the original source.")
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"Original narration audio is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".tmp.wav")
    try:
        _run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", str(source), "-vn", "-af", build_filter(settings),
            "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(temporary),
        ])
        if not temporary.is_file() or temporary.stat().st_size < 44:
            raise RuntimeError("FFmpeg did not produce a valid processed WAV.")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def join_audio_segments(segments, destination, pause_ms=250, ffmpeg="ffmpeg"):
    """Convert speech segments to a common format and insert sentence pauses."""
    segments = [Path(item).resolve() for item in segments]
    destination = Path(destination).resolve()
    if not segments:
        raise ValueError("At least one narration segment is required.")
    if any(not item.is_file() or item.stat().st_size == 0 for item in segments):
        raise FileNotFoundError("One or more original narration segments are missing.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pause = max(0, min(800, int(pause_ms))) / 1000.0
    temporary_dir = destination.parent / "_join_tmp"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    try:
        normalized = []
        for index, source in enumerate(segments):
            target = temporary_dir / f"segment_{index:03d}.wav"
            _run([
                ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                "-i", str(source), "-vn", "-ar", "48000", "-ac", "2",
                "-c:a", "pcm_s16le", str(target),
            ])
            normalized.append(target)
        if len(normalized) == 1 or pause <= 0:
            listing = temporary_dir / "segments.txt"
            listing.write_text(
                "".join(f"file '{item.as_posix()}'\n" for item in normalized),
                encoding="utf-8",
            )
            _run([
                ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                "-f", "concat", "-safe", "0", "-i", str(listing),
                "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", str(destination),
            ])
        else:
            inputs = []
            for item in normalized:
                inputs.extend(["-i", str(item)])
            for _ in range(len(normalized) - 1):
                inputs.extend([
                    "-f", "lavfi", "-t", f"{pause:.3f}", "-i",
                    "anullsrc=r=48000:cl=stereo",
                ])
            filters = []
            ordered = []
            count = len(normalized)
            for index in range(count):
                filters.append(f"[{index}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a{index}]")
                ordered.append(f"[a{index}]")
                if index < count - 1:
                    silence_index = count + index
                    filters.append(f"[{silence_index}:a]atrim=duration={pause:.3f},asetpts=PTS-STARTPTS[s{index}]")
                    ordered.append(f"[s{index}]")
            filters.append("".join(ordered) + f"concat=n={len(ordered)}:v=0:a=1[out]")
            _run([
                ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                *inputs, "-filter_complex", ";".join(filters), "-map", "[out]",
                "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(destination),
            ])
        if not destination.is_file() or destination.stat().st_size < 44:
            raise RuntimeError("Could not join narration segments.")
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)
    return destination
