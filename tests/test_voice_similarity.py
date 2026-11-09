from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


def _embedder():
    from backend.voice import similarity
    embedder=similarity.load(ROOT)
    if embedder is None:pytest.skip('缺少 campplus 或 CPU 依赖')
    return similarity,embedder


def _tone(rate,starts,seconds=3.):
    t=np.arange(int(rate*seconds))/rate
    return sum(np.sin(2*np.pi*f*t)/index for index,f in enumerate(starts,1)).astype(np.float32)*.1


def test_embedding_is_normalised_and_repeatable():
    similarity,embedder=_embedder()
    rate=24000;signal=_tone(rate,[150,300,450])
    first=embedder.embedding(signal,rate);second=embedder.embedding(signal,rate)
    assert first.shape==(192,)
    assert float(np.linalg.norm(first))==pytest.approx(1.,abs=1e-5)
    assert similarity.cosine(first,second)==pytest.approx(1.,abs=1e-6)


def test_different_pitch_landscape_scores_lower_than_itself():
    similarity,embedder=_embedder()
    rate=24000
    left=embedder.embedding(_tone(rate,[150,300,450]),rate)
    right=embedder.embedding(_tone(rate,[220,440,660]),rate)
    assert similarity.cosine(left,left)>similarity.cosine(left,right)
