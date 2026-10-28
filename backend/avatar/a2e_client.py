"""Shared WAV request and safe NPZ decoding for the local expression service."""
import io
import httpx
import numpy as np


async def request_weights(audio: bytes, *, timeout=180, headers=None):
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        response = await client.post('http://127.0.0.1:9881/expression', content=audio, headers=headers)
        response.raise_for_status()
    with np.load(io.BytesIO(response.content), allow_pickle=False) as data:
        return data['weights']
