from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class OnlineState:
    pop: Dict[int, Tuple[np.ndarray, float]]
    q: Dict[int, Dict[float, Tuple[np.ndarray, float]]]
    trained_samples: int = 0


def sgd_logistic_step(
    w: np.ndarray,
    b: float,
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    l2: float,
    sample_weight: np.ndarray | None = None,
    trainable: np.ndarray | None = None,
) -> Tuple[np.ndarray, float]:
    n = max(1, len(y))
    z = X @ w + b
    p = sigmoid(z)
    if sample_weight is None:
        grad = p - y
    else:
        grad = sample_weight * (p - y)

    # Scale the loss gradient by the number of samples but keep the
    # regularizer an absolute per-step penalty. Dividing l2 by n made the
    # decay 8% per update when only one sample was available, which wiped out
    # everything the model learned between training runs.
    gw = (X.T @ grad) / n + l2 * w
    gb = float(np.mean(grad))
    if trainable is not None:
        # Freeze the globally-trained base coefficients and the intercept. A
        # local hourly batch is a few hundred autocorrelated minutes; letting it
        # move all 38 base weights drifts them by ~0.1 per run, and letting it
        # move the intercept shifts the whole probability scale until everything
        # looks rainy.
        gw = gw * trainable
        gb = 0.0
    return w - lr * gw, b - lr * gb


def sgd_quantile_step(
    w: np.ndarray,
    b: float,
    X: np.ndarray,
    y: np.ndarray,
    q: float,
    lr: float,
    l2: float,
    trainable: np.ndarray | None = None,
) -> Tuple[np.ndarray, float]:
    n = max(1, len(y))
    pred = X @ w + b
    e = y - pred
    grad = np.where(e >= 0, -q, 1 - q)  # dL/dpred
    gw = (X.T @ grad) / n + l2 * w
    gb = float(np.mean(grad))
    if trainable is not None:
        gw = gw * trainable
        gb = 0.0
    return w - lr * gw, b - lr * gb


def apply_training(
    state: OnlineState,
    batch: Dict[int, Dict[str, Any]],
    lr_pop: float = 0.05,
    lr_q: float = 0.02,
    l2: float = 1e-3,
    pos_weight: float = 12.0,
    epochs: int = 40,
    trainable: np.ndarray | None = None,
    max_trainable_weight: float = 6.0,
) -> Dict[str, Any]:
    """batch[h] = {'X': [x...], 'amt': [mm...], 'pop': [0/1...]} with x already normalized"""
    info: Dict[str, Any] = {"updated": 0, "by_horizon": {}}

    def _cap(w: np.ndarray) -> np.ndarray:
        # The trainable coordinates are the locally-adapted pre-rain
        # indicators. l2 keeps them from running away, but with the same batch
        # replayed across consecutive runs the norm still creeps upward. A hard
        # cap guarantees the weights can neither explode nor stay unbounded over
        # a long-running process.
        if trainable is None:
            return w
        excess = np.abs(w) > max_trainable_weight
        if excess.any():
            w = w.copy()
            w[excess] = np.sign(w[excess]) * max_trainable_weight
        return w

    for h, data in batch.items():
        X = np.asarray(data["X"], dtype=float)
        if X.size == 0:
            continue
        y_pop = np.asarray(data["pop"], dtype=float)
        y_amt = np.asarray(data["amt"], dtype=float)

        sample_weight = np.ones_like(y_pop, dtype=float)
        sample_weight[y_pop > 0.5] = pos_weight

        if h not in state.pop:
            continue

        # Several passes over the batch. A single pass per hourly run cannot
        # move the weights far enough to matter.
        w, b = state.pop[h]
        for _ in range(max(1, epochs)):
            w, b = sgd_logistic_step(
                w, b, X, y_pop, lr=lr_pop, l2=l2,
                sample_weight=sample_weight, trainable=trainable,
            )
            w = _cap(w)
        state.pop[h] = (w, b)

        # Quantiles on log1p
        y_log = np.log1p(np.maximum(0.0, y_amt))
        for q in (0.1, 0.5, 0.9):
            if q not in state.q.get(h, {}):
                continue
            wq, bq = state.q[h][q]
            for _ in range(max(1, epochs)):
                wq, bq = sgd_quantile_step(
                    wq, bq, X, y_log, q=q, lr=lr_q, l2=l2,
                    trainable=trainable,
                )
            state.q[h][q] = (wq, bq)

        state.trained_samples += int(len(y_pop))
        info["updated"] += int(len(y_pop))
        info["by_horizon"][str(h)] = int(len(y_pop))
    return info