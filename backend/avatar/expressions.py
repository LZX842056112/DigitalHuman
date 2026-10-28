from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class ExpressionMap:
    coefficients: np.ndarray
    jaw_coefficients: np.ndarray

    @classmethod
    def fit(cls, flame_basis, arkit_basis, jaw_basis=None, ridge: float = 1e-5):
        flame = np.asarray(flame_basis, dtype=np.float64)
        arkit = np.asarray(arkit_basis, dtype=np.float64)
        if flame.ndim != 2 or arkit.ndim != 2 or flame.shape[0] != arkit.shape[0]:
            raise ValueError("表情模板的共同顶点数量不一致")
        if not np.isfinite(flame).all() or not np.isfinite(arkit).all() or np.linalg.norm(flame) == 0:
            raise ValueError("表情模板无有效顶点位移")
        jaw = np.zeros((flame.shape[0], 3)) if jaw_basis is None else np.asarray(jaw_basis)
        if jaw.shape != (flame.shape[0], 3):
            raise ValueError("下颌模板的共同顶点数量不一致")
        design = np.concatenate([flame, jaw], axis=1)
        if ridge:
            design_aug = np.concatenate([design, np.eye(design.shape[1]) * np.sqrt(ridge)])
            target = np.concatenate([arkit, np.zeros((design.shape[1], arkit.shape[1]))])
        else:
            design_aug, target = design, arkit
        fitted = np.linalg.lstsq(design_aug, target, rcond=None)[0]
        return cls(fitted[:flame.shape[1]].astype(np.float32), fitted[flame.shape[1]:].astype(np.float32))

    def apply(self, weights):
        weights = np.asarray(weights, dtype=np.float32)
        if weights.shape[-1] != self.coefficients.shape[1] or not np.isfinite(weights).all():
            raise ValueError("表情系数数量或数值无效")
        weights = np.clip(weights, 0, 1)
        expression = np.clip(weights @ self.coefficients.T, -3, 3)
        jaw = np.clip(weights @ self.jaw_coefficients.T, -.5, .5)
        return expression, jaw

    def save(self, path: Path):
        np.savez_compressed(path, version=np.array([1]), coefficients=self.coefficients, jaw=self.jaw_coefficients)

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as data:
            return cls(data["coefficients"], data["jaw"])
