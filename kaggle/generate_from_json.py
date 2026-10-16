"""Generate a Manual Timeline JSON project without opening Streamlit."""
from __future__ import annotations
import argparse
import ast
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "webui" / "pages" / "2_Manual_Timeline.py"
OUT = ROOT / "storage" / "manual_timeline"


def load_render_namespace():
    imports, functions = [], []
    tree = ast.parse(PAGE.read_text(encoding="utf-8"), filename=str(PAGE))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.decorator_list = []
            functions.append(node)
    module = ast.Module(body=imports + functions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"__file__": str(PAGE), "__name__": "mpt_manual_timeline_cli", "ROOT": ROOT, "OUT": OUT}
    exec(compile(module, str(PAGE), "exec"), namespace)
    namespace.update(ROOT=ROOT, OUT=OUT)
    return namespace


def configure_indicf5_profiles(project):
    api_url = os.environ.get("INDICF5_API_URL", "http://127.0.0.1:8765").rstrip("/")
    reference_audio = os.environ.get("INDICF5_REFERENCE_AUDIO", "").strip()
    reference_text = os.environ.get("INDICF5_REFERENCE_TEXT", "").strip()
    voices = [str(project.get("settings", {}).get("voice", ""))]
    voices.extend(str(s.get("voice") or "") for s in project.get("scenes", []))
    indic_voices = [v for v in voices if v.startswith("indicf5:")]
    if not indic_voices:
        return
    if not reference_audio or not Path(reference_audio).is_file() or not reference_text:
        raise RuntimeError("IndicF5 scenes require INDICF5_REFERENCE_AUDIO (WAV path) and INDICF5_REFERENCE_TEXT (exact transcript).")
    profile_dir = ROOT / "storage" / "indicf5_profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    for voice in indic_voices:
        profile_id = voice.split(":", 1)[1].strip()
        if not profile_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in profile_id.lower()):
            raise ValueError(f"Invalid IndicF5 profile id in voice field: {voice}")
        data = {"id": profile_id, "name": "Kaggle IndicF5",
                "reference_audio": str(Path(reference_audio).resolve()),
                "reference_text": reference_text, "api_url": api_url}
        (profile_dir / f"{profile_id}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", help="Manual Timeline project JSON")
    parser.add_argument("--output-dir", help="Project working directory")
    parser.add_argument("--cleanup", action="store_true", help="Remove intermediates after a successful render, keeping final MP4")
    args = parser.parse_args()
    json_path = Path(args.json_file).expanduser().resolve()
    if not json_path.is_file():
        parser.error(f"JSON file not found: {json_path}")
    project = json.loads(json_path.read_text(encoding="utf-8"))
    ns = load_render_namespace()
    project = ns["_validate_imported_project"](project)
    configure_indicf5_profiles(project)
    project_id = "".join(c if c.isalnum() or c in "_-" else "_" for c in str(project.get("project_id") or "my_video"))
    folder = Path(args.output_dir).expanduser().resolve() if args.output_dir else OUT / project_id
    folder.mkdir(parents=True, exist_ok=True)

    def progress(percent, message):
        print(f"[{percent:3d}%] {message}", flush=True)

    output = Path(ns["render"](project, folder, progress_callback=progress)).resolve()
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("Render completed but final MP4 is missing or empty.")
    print(f"FINAL_VIDEO={output}", flush=True)
    if args.cleanup:
        removed = ns["_cleanup_generated_temp_files"](folder, output)
        print(f"CLEANUP_REMOVED_ITEMS={removed}", flush=True)


if __name__ == "__main__":
    main()
