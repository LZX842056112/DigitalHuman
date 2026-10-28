"""CPU-only adapter around the published LAM-A2E network."""
import io
import os
import sys
import threading
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
from scipy.signal import resample_poly
from math import gcd

ROOT = Path(os.environ["DIGITAL_HUMAN_ROOT"]).resolve()
app = FastAPI(title="Local audio-to-expression")
_model = None
_lock = threading.Lock()


def load_cpu_model():
    from backend.storage import read_json
    repo = ROOT / "third_party/a2e"
    sys.path.insert(0, str(repo))
    from models.network import Audio2Expression
    index = read_json(ROOT / "models/assets-index.json")
    model = Audio2Expression(device=torch.device("cpu"), pretrained_encoder_type="wav2vec",
            pretrained_encoder_path=str(ROOT / "models/wav2vec2"),
            wav2vec2_config_path=str(repo / "configs/wav2vec2_config.json"),
            num_identity_classes=12, identity_feat_dim=64, hidden_dim=512,
            expression_dim=52, norm_type="ln", use_transformer=False)
    checkpoint = torch.load(index["a2e_weights"], map_location="cpu", weights_only=False)
    state = {k.removeprefix("module.").removeprefix("backbone."): v for k,v in checkpoint["state_dict"].items()
             if not k.removeprefix("module.").startswith("criteria.")}
    model.load_state_dict(state, strict=True)
    model.cpu().eval().requires_grad_(False)
    torch.set_num_threads(4)
    return model


@app.get("/health")
def health():
    return {"device": "cpu", "loaded": _model is not None}


@app.post("/expression")
async def expression(request: Request):
    global _model
    data = await request.body()
    if len(data) > 32 * 1024**2:
        raise HTTPException(413, "Audio is too large")
    pcm, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    pcm = pcm.mean(axis=1)
    if rate != 16000:
        divisor = gcd(rate, 16000)
        pcm = resample_poly(pcm, 16000 // divisor, rate // divisor).astype(np.float32)
    if len(pcm) < 1600:
        raise HTTPException(400, "Audio is too short")
    with _lock, torch.inference_mode():
        if _model is None:
            _model = load_cpu_model()
        identity = torch.nn.functional.one_hot(torch.tensor(0), 12).float()[None]
        weights = _model({"input_audio_array": torch.from_numpy(pcm)[None], "id_idx": identity})[0].cpu().numpy()
    result = io.BytesIO()
    np.savez_compressed(result, weights=weights.astype(np.float32))
    return Response(result.getvalue(), media_type="application/octet-stream")
