from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

BASE_MODEL_PATH = Path("/app/model/base_model.json")

# clip to keep feature vector stable even if some normalization stats are placeholders
Z_CLIP = 10.0


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        try:
            f = float(v)
            return f if np.isfinite(f) else None
        except Exception:
            return None
    try:
        s = str(v).strip()
        num = ""
        dot = False
        sign = False
        for ch in s:
            if ch in "+-" and not sign and not num:
                num += ch
                sign = True
            elif ch.isdigit():
                num += ch
            elif ch == "." and not dot:
                num += ch
                dot = True
            elif num:
                break
        if not num:
            return None
        f = float(num)
        return f if np.isfinite(f) else None
    except Exception:
        return None


def _sanitize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    bad = ~np.isfinite(x)
    if bad.any():
        x = x.copy()
        x[bad] = 0.0
    return x


@dataclass
class BaseModel:
    version: str
    features: List[str]
    pop_models: Dict[str, Dict[str, Any]]
    amount_models: Dict[str, Dict[str, Any]]
    # per-feature normalization: {feature: {mean: float, std: float}}
    norm: Dict[str, Dict[str, float]]

    @classmethod
    def load(cls, path: Path = BASE_MODEL_PATH) -> "BaseModel":
        obj = json.loads(path.read_text(encoding="utf-8"))
        features = list(obj.get("features", []))

        norm: Dict[str, Dict[str, float]] = {}

        normalization = obj.get("normalization")
        if isinstance(normalization, dict):
            mean = normalization.get("mean")
            std = normalization.get("std")
            if (
                isinstance(mean, list)
                and isinstance(std, list)
                and len(mean) == len(features)
                and len(std) == len(features)
            ):
                for i, fname in enumerate(features):
                    mu = float(mean[i])
                    sd_raw = float(std[i])
                    sd = sd_raw if sd_raw != 0.0 else 1.0
                    norm[fname] = {"mean": mu, "std": sd}

        if not norm and isinstance(obj.get("norm"), dict):
            norm = dict(obj.get("norm", {}))

        return cls(
            version=str(obj.get("version", "unknown")),
            features=features,
            pop_models=dict(obj.get("pop_models", {})),
            amount_models=dict(obj.get("amount_models", {})),
            norm=norm,
        )

    def featurize(self, row: Dict[str, Any]) -> np.ndarray:
        vals: List[float] = []
        for name in self.features:
            v = _to_float(row.get(name))
            st = self.norm.get(name)

            if v is None or not st:
                vals.append(0.0)
                continue

            mu = float(st.get("mean", 0.0))
            sd = float(st.get("std", 1.0)) or 1.0

            z = (v - mu) / sd
            if not np.isfinite(z):
                z = 0.0

            # critical safety: clip z so placeholders (mean=0,std=1) can't explode the model
            if z > Z_CLIP:
                z = Z_CLIP
            elif z < -Z_CLIP:
                z = -Z_CLIP

            vals.append(float(z))

        return _sanitize(np.array(vals, dtype=float))
