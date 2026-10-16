"""Kaggle-compatible IndicF5 HTTP API for MoneyPrinterTurbo."""
from __future__ import annotations
import io
import os
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from transformers import AutoModel

app = FastAPI(title="IndicF5 API", version="kaggle-1.0")
model = None
model_device = "unloaded"
model_error: Optional[str] = None


class GenerateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=12000)
    reference_audio: Optional[str] = None
    reference_text: Optional[str] = None


def _select_device() -> str:
    requested = os.environ.get("INDICF5_DEVICE", "auto").lower()
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("INDICF5_DEVICE requests CUDA but CUDA is not available.")
    return requested


@app.on_event("startup")
def load_model() -> None:
    global model, model_device, model_error
    try:
        device = _select_device()
        model_id = os.environ.get("INDICF5_MODEL_ID", "ai4bharat/IndicF5")
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True, token=token or None)
        if hasattr(model, "to"):
            model = model.to(device)
        if hasattr(model, "eval"):
            model.eval()
        model_device = device
        model_error = None
        print(f"[IndicF5] loaded {model_id} on {model_device}", flush=True)
    except Exception as exc:
        model = None
        model_device = "failed"
        model_error = f"{type(exc).__name__}: {exc}"
        print(f"[IndicF5] model load failed: {model_error}", flush=True)


@app.get("/health")
def health():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "ok": model is not None,
        "model_id": os.environ.get("INDICF5_MODEL_ID", "ai4bharat/IndicF5"),
        "device": model_device,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "error": model_error,
    }


@app.post("/generate")
def generate(payload: GenerateRequest):
    if model is None:
        raise HTTPException(status_code=503, detail=f"IndicF5 model is not loaded: {model_error}")
    text = payload.text.strip()
    reference_audio = (payload.reference_audio or os.environ.get("INDICF5_REFERENCE_AUDIO") or "").strip()
    reference_text = (payload.reference_text or os.environ.get("INDICF5_REFERENCE_TEXT") or "").strip()
    if not reference_audio or not Path(reference_audio).is_file():
        raise HTTPException(status_code=400, detail="A valid reference_audio WAV path is required.")
    if not reference_text:
        raise HTTPException(status_code=400, detail="reference_text must match the reference audio.")
    try:
        with torch.inference_mode():
            audio = model(text, ref_audio_path=reference_audio, ref_text=reference_text)
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        audio = np.asarray(audio)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        else:
            audio = audio.astype(np.float32, copy=False)
        audio = np.squeeze(audio)
        if audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError("model returned empty or non-finite audio")
        buffer = io.BytesIO()
        sf.write(buffer, audio, samplerate=24000, format="WAV", subtype="PCM_16")
        return Response(content=buffer.getvalue(), media_type="audio/wav")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("INDICF5_HOST", "127.0.0.1"),
                port=int(os.environ.get("INDICF5_PORT", "8765")), log_level="info")
