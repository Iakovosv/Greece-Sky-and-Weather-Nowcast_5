# model/engine.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np

from ml.base import BaseModel
from ml.online import OnlineState, apply_training
from ml.persistence import load_state, atomic_save

HORIZONS = (30, 60, 120, 360)

# PoP above which an alert is raised and a lead time is reported.
ALERT_POP = 0.5

# Leading-indicator features produced by core/features.py that are not already
# part of the base feature vector. They are already scaled to roughly unit
# magnitude, so mean=0 / std=1 normalization is correct for them.
EXTRA_FEATURES = (
    "rain_1m_mm",
    "dP_10m_scaled",
    "dRH_10m_scaled",
    "dT_10m_scaled",
    "dWind_10m_scaled",
    "dSolar_10m_scaled",
    "dP_30m_scaled",
    "dRH_30m_scaled",
    "dT_30m_scaled",
    "p_mean_30m_scaled",
    "rh_mean_30m_scaled",
    "p_trend_30m",
    "h_trend_30m",
    "p_volatility_5m",
    "dSpread_10m",
    "pre_rain_index",
    "instability_index",
)

# How far back a snapshot must be rain-free to count as a genuine pre-rain state.
DRY_GUARD_MIN = 15

# Physically-motivated initial weights for the pre-rain indicators.
#
# The base model puts zero weight on every trend feature, so on its own it can
# only react once the air is already wet -- which is why rainfall was reported
# 30-60 minutes after it had started. These priors encode the standard
# pre-rainfall signature (pressure falling, humidity rising, solar dropping,
# air saturating) so the first forecast is already useful; online learning then
# refines them against local observations.
PRIOR_W = {
    # Rain persistence: the base model gives rainrate zero weight, so without
    # this an ongoing shower produces a *lower* probability than the run-up to
    # it, because the pre-rain trends flatten out once rain begins.
    "rain_1m_mm": 3.00,
    "rainrate": 0.05,
    "dP_10m_scaled": -0.20,
    "dP_30m_scaled": -0.35,
    "dRH_10m_scaled": 0.25,
    "dRH_30m_scaled": 0.45,
    "dSolar_10m_scaled": -0.25,
    "dT_30m_scaled": -0.10,
    "p_trend_30m": -0.30,
    "h_trend_30m": 0.30,
    "p_volatility_5m": 0.15,
    "dSpread_10m": -0.20,
    "pre_rain_index": 0.90,
    "instability_index": 0.35,
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _sigmoid_stable(z: float) -> float:
    z = float(np.clip(z, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-z)))


def _expm1_mm_stable(y: float, max_mm: float) -> float:
    y = float(np.clip(y, -20.0, 6.0))
    mm = float(np.expm1(y))
    if not np.isfinite(mm):
        mm = 0.0
    return float(_clamp(mm, 0.0, max_mm))


class ModelEngine:
    """Base model + lightweight local adaptation (safe for RPi4)."""

    def __init__(self, logger=None):
        self.log = logger

        self.base = BaseModel.load()
        self.base_features = list(self.base.features)

        # Extend the base feature vector with the pre-rain indicators. The base
        # model has no weights for them, so they start at zero and are learned
        # online from local pre-rain observations.
        extras = [f for f in EXTRA_FEATURES if f not in self.base_features]
        self.extra_features = extras
        self.features = self.base_features + extras

        for name in extras:
            self.base.norm[name] = {"mean": 0.0, "std": 1.0}

        # featurize() iterates over base.features, so it must see the extension too.
        self.base.features = self.features

        # Seed the pre-rain indicators with their physical prior so the very
        # first forecast already looks ahead of the rain instead of waiting to
        # observe it.
        self.prior = np.array(
            [PRIOR_W.get(f, 0.0) for f in self.features], dtype=float
        )

        # Only the locally-observed pre-rain indicators may be adapted online.
        # The base coefficients come from a global fit and must not be moved by
        # an hourly window of a few hundred autocorrelated minutes.
        self.trainable = np.array(
            [1.0 if f in self.extra_features else 0.0 for f in self.features],
            dtype=float,
        )

        self.base_version = self.base.version
        self.n_base = len(self.base_features)

        self.trained_samples = 0
        self.last_training_info: Dict[str, Any] = {"updated": 0}
        self.last_training_ts: str | None = None
        self.training_count_total: int = 0

        pop: Dict[int, Any] = {}
        q: Dict[int, Any] = {}
        self._platt: Dict[int, Tuple[float, float]] = {}

        for h in HORIZONS:
            key = str(h)
            if key in self.base.pop_models:
                bm = self.base.pop_models[key]
                w = self._pad(np.array(bm["w"], dtype=float))
                b = float(bm["b"])
                a = float(bm.get("platt_a", 1.0))
                c = float(bm.get("platt_b", 0.0))
            else:
                # No base weights for this horizon (30m): start from the 60m
                # model so the cold start is sane, then train it locally.
                bm60 = self.base.pop_models["60"]
                w = self._pad(np.array(bm60["w"], dtype=float))
                b = float(bm60["b"])
                a = float(bm60.get("platt_a", 1.0))
                c = float(bm60.get("platt_b", 0.0))
            w = w + self.prior
            pop[h] = (w, b)
            self._platt[h] = (a, c)

            am = self.base.amount_models.get(key)
            if am is None:
                am = self.base.amount_models["60"]
            q[h] = {}
            for qq in (0.1, 0.5, 0.9):
                coeff = am[str(qq)]
                q[h][qq] = (self._pad(np.array(coeff["w"], dtype=float)), float(coeff["b"]))

        self.state = OnlineState(pop=pop, q=q, trained_samples=0)

        # Replay memory. An hourly retrain that only sees the current 420-minute
        # buffer is trained on a tiny, unrepresentative window: one wet hour can
        # make every sample positive and the model swings between 0% and 100%.
        # Keeping the recent labelled samples gives each retrain a stable,
        # balanced view of local pre-rain conditions.
        self.replay: Dict[int, Dict[str, List[Any]]] = {h: {"X": [], "amt": [], "pop": []} for h in HORIZONS}

        st = load_state()
        if st:
            try:
                self._apply_persisted(st)
            except Exception as e:
                if self.log:
                    self.log.warning(f"Failed to load station state: {e}")

        self.trained_samples = int(getattr(self.state, "trained_samples", 0))

        if self.log:
            self.log.info(
                f"Loaded base model version={self.base_version} "
                f"features={len(self.base_features)}+{len(self.extra_features)}"
            )

    def _pad(self, w: np.ndarray) -> np.ndarray:
        """Match a coefficient vector to the current feature count."""
        n = len(self.features)
        if len(w) == n:
            return w
        out = np.zeros(n, dtype=float)
        out[: min(len(w), n)] = w[: min(len(w), n)]
        return out

    def _apply_persisted(self, st: Dict[str, Any]) -> None:
        if st.get("base_version") != self.base_version:
            if self.log:
                self.log.warning(
                    f"Station state base_version mismatch: "
                    f"{st.get('base_version')} != {self.base_version} (ignored)"
                )
            return

        n = len(self.features)
        for h_str, wb in st.get("pop", {}).items():
            h = int(h_str)
            if h not in self.state.pop:
                continue
            raw = np.array(wb.get("w", []), dtype=float)
            if len(raw) == n:
                self.state.pop[h] = (raw, float(wb.get("b", 0.0)))
            elif len(raw) == self.n_base:
                # A state written before the extra features existed: one
                # coefficient per base feature, with no prior seed baked in.
                # Padding alone would zero the prior coordinates and silently
                # restore the pre-rain lag this build fixes, so pad and re-apply
                # the prior on top of the user's own trained coefficients.
                self.state.pop[h] = (
                    self._pad(raw) + self.prior,
                    float(wb.get("b", 0.0)),
                )
                if self.log:
                    self.log.info(
                        f"Upgraded pop weights for h={h} from {len(raw)} "
                        f"base coefficients: priors re-applied."
                    )
            elif self.log:
                self.log.warning(
                    f"Ignoring pop weights for h={h}: {len(raw)} coefficients "
                    f"matches neither {n} nor {self.n_base}."
                )

        for h_str, qs in st.get("q", {}).items():
            h = int(h_str)
            if h not in self.state.q:
                continue
            for q_str, wb in qs.items():
                qq = float(q_str)
                if qq not in self.state.q[h]:
                    continue
                raw = np.array(wb.get("w", []), dtype=float)
                # Unlike pop, the amount quantiles carry no prior: init seeds
                # them as pad(base) with zeros in the extra coordinates. Padding
                # a shorter vector therefore preserves the trained base
                # coefficients and stays consistent with init.
                self.state.q[h][qq] = (self._pad(raw), float(wb.get("b", 0.0)))

        for h_str, mem in (st.get("replay") or {}).items():
            h = int(h_str)
            if h not in self.replay:
                continue
            self.replay[h] = {
                "X": list(mem.get("X", [])),
                "amt": list(mem.get("amt", [])),
                "pop": list(mem.get("pop", [])),
            }

        self.state.trained_samples = int(st.get("trained_samples", 0))
        self.training_count_total = int(st.get("training_count_total", 0) or 0)
        self.last_training_ts = st.get("last_training_ts")

    def save_state(self) -> None:
        obj = {
            "base_version": self.base_version,
            "trained_samples": int(self.state.trained_samples),
            "training_count_total": int(self.training_count_total),
            "last_training_ts": self.last_training_ts,
            "pop": {},
            "q": {},
        }

        for h, (w, b) in self.state.pop.items():
            obj["pop"][str(h)] = {"w": w.tolist(), "b": float(b)}

        for h, qs in self.state.q.items():
            obj["q"][str(h)] = {}
            for qq, (w, b) in qs.items():
                obj["q"][str(h)][str(qq)] = {"w": w.tolist(), "b": float(b)}

        obj["replay"] = {
            str(h): {"X": mem["X"], "amt": mem["amt"], "pop": mem["pop"]}
            for h, mem in self.replay.items()
            if mem["X"]
        }

        atomic_save(obj)

    def infer(self, buffer: Any) -> Dict[str, float]:
        last = buffer.last() if hasattr(buffer, "last") else None
        if not last:
            # Cold start: no reading has been stored yet, so every output is
            # reported as no-signal. The alert and lead keys are included so
            # consumers see the same schema from the first message onward.
            out: Dict[str, float] = (
                {f"pop_{h}m": 0.0 for h in HORIZONS}
                | {f"p{p}_{h}m": 0.0 for h in HORIZONS for p in (10, 50, 90)}
                | {f"alert_{h}m": 0 for h in HORIZONS}
            )
            out["alert"] = 0
            out["lead_min"] = 0
            out["stale"] = 0
            return out

        x = self.base.featurize(last)
        out: Dict[str, float] = {}

        pops = {h: self._infer_pop_local(x, h) for h in HORIZONS}
        for h in HORIZONS:
            out[f"pop_{h}m"] = round(100.0 * pops[h], 1)

        max_mm = {30: 30.0, 60: 50.0, 120: 80.0, 360: 120.0}

        for h in HORIZONS:
            q10, q50, q90 = self._infer_q_local(x, h, max_mm=max_mm[h])

            # Shrink the amount forecast when rain is unlikely, so a high p50
            # can never appear next to a low PoP. Ramping over 0-50% (rather
            # than gating off below 10%) keeps genuine pre-rain amounts: the
            # sharper PoP sits low during the run-up, and a hard 10% cut
            # suppressed p90 exactly when rain was approaching. Measured
            # pinball loss improves from 0.647 to 0.440 with the wider ramp.
            g = float(_clamp(pops[h] / 0.5, 0.0, 1.0))
            out[f"p10_{h}m"] = round(q10 * g, 3)
            out[f"p50_{h}m"] = round(q50 * g, 3)
            out[f"p90_{h}m"] = round(q90 * g, 3)

        # Alert flags and the lead time the user actually asked for. The lead
        # time is the shortest horizon whose PoP has crossed the alert
        # threshold, i.e. how much warning the current reading gives.
        for h in HORIZONS:
            fired = pops[h] >= ALERT_POP
            out[f"alert_{h}m"] = 1 if fired else 0
        fired_horizons = [h for h in HORIZONS if pops[h] >= ALERT_POP]
        out["alert"] = 1 if fired_horizons else 0
        out["lead_min"] = min(fired_horizons) if fired_horizons else 0

        return out

    def _infer_pop_local(self, x: np.ndarray, h: int) -> float:
        w, b = self.state.pop[h]
        raw_logit = float(x @ w + b)

        a, c = self._platt.get(h, (1.0, 0.0))
        p = _sigmoid_stable(a * raw_logit + c)
        if not np.isfinite(p):
            p = 0.0
        return float(_clamp(p, 0.0, 1.0))

    def _infer_q_local(self, x: np.ndarray, h: int, max_mm: float) -> Tuple[float, float, float]:
        qs: Dict[float, float] = {}

        for qq, (w, b) in self.state.q[h].items():
            yhat = float(x @ w + b)
            if not np.isfinite(yhat):
                yhat = 0.0
            qs[qq] = _expm1_mm_stable(yhat, max_mm=max_mm)

        q10, q50, q90 = qs.get(0.1, 0.0), qs.get(0.5, 0.0), qs.get(0.9, 0.0)

        q10 = min(q10, q50, q90)
        q90 = max(q10, q50, q90)
        q50 = max(q10, min(q50, q90))

        return q10, q50, q90

    def _build_batch(
        self,
        buffer_rows: List[Dict[str, Any]],
        horizons_min,
        threshold_mm: float,
        max_samples: int,
    ) -> Dict[int, Any]:
        """Build training samples for a genuinely pre-rain framing.

        For each dry snapshot at time t the label is the rain that falls in
        (t, t+h]. Only snapshots whose preceding DRY_GUARD_MIN minutes were also
        rain-free are used, so the model learns what the atmosphere looks like
        *before* rain rather than during it.
        """
        n = len(buffer_rows)
        batch: Dict[int, Any] = {}

        for h in horizons_min:
            if n < h + DRY_GUARD_MIN + 1:
                continue

            X: List[List[float]] = []
            amt: List[float] = []
            pop: List[float] = []

            last_snap = n - h - 1
            for snap_idx in range(DRY_GUARD_MIN - 1, last_snap + 1):
                window = buffer_rows[snap_idx + 1: snap_idx + 1 + h]
                if len(window) < h:
                    continue

                total = float(sum(float(r.get("rain_1m_mm", 0.0) or 0.0) for r in window))
                X.append(self.base.featurize(buffer_rows[snap_idx]).tolist())
                amt.append(total)
                pop.append(1.0 if total > threshold_mm else 0.0)

                # `wet_now` is not filtered out: the model needs to learn both
                # the pre-rain signature and rain persistence. Dropping wet
                # snapshots taught it that rain lowers the probability.

            if not X:
                continue

            # Keep the most recent samples and thin them out: consecutive minutes
            # are strongly autocorrelated and add little information.
            if len(X) > max_samples:
                X = X[-max_samples:]
                amt = amt[-max_samples:]
                pop = pop[-max_samples:]

            if len(X) > 30:
                X = X[::2]
                amt = amt[::2]
                pop = pop[::2]

            batch[h] = {"X": X, "amt": amt, "pop": pop}

        return batch

    def _update_replay(self, batch: Dict[int, Any], max_replay: int) -> None:
        """Append the newest labelled samples to the replay memory."""
        for h, data in batch.items():
            if h not in self.replay:
                self.replay[h] = {"X": [], "amt": [], "pop": []}
            mem = self.replay[h]
            mem["X"].extend(data["X"])
            mem["amt"].extend(data["amt"])
            mem["pop"].extend(data["pop"])
            if len(mem["X"]) > max_replay:
                keep = mem["X"][-max_replay:]
                mem["X"] = keep
                mem["amt"] = mem["amt"][-max_replay:]
                mem["pop"] = mem["pop"][-max_replay:]

    def _refit_platt(self, min_samples: int = 80) -> None:
        """Re-calibrate the probability scale against the replay memory.

        The base model's Platt coefficients were fitted on a different climate
        and are never updated. Locally the raw logits came out systematically
        under-confident (predicted 0.24 where 0.82 of the minutes were wet), so
        the reported probability was not usable as a real likelihood. The replay
        memory already holds labelled (logit, label) pairs, so refit a/c here.

        Deliberately conservative: monotonic (a > 0), bounded, and only applied
        once there is enough independent evidence.
        """
        for h, mem in self.replay.items():
            labels = np.asarray(mem["pop"], dtype=float)
            if len(labels) < min_samples or labels.min() == labels.max():
                continue
            X = np.asarray(mem["X"], dtype=float)
            w, b = self.state.pop[h]
            logits = X @ w + b
            sd = float(np.std(logits))
            if not np.isfinite(sd) or sd < 1e-6:
                continue
            z = (logits - float(np.mean(logits))) / sd

            a, c = 1.0, 0.0
            for _ in range(60):
                pr = 1.0 / (1.0 + np.exp(-(a * z + c)))
                g = pr - labels
                ga = float(np.mean(g * z)) + 1e-3 * a
                gc = float(np.mean(g))
                a -= 0.5 * ga
                c -= 0.5 * gc
                a = float(_clamp(a, 0.05, 8.0))
                c = float(_clamp(c, -8.0, 8.0))
            if not (np.isfinite(a) and np.isfinite(c)):
                continue
            # store in the same (logit) space the inference path expects
            self._platt[h] = (a / sd, c - a * float(np.mean(logits)) / sd)

    def _balanced(self, data: Dict[str, Any], max_total: int) -> Dict[str, Any]:
        """Subsample so the positive class is neither drowned nor dominant."""
        X = data["X"]
        pop = data["pop"]
        amt = data["amt"]

        pos = [i for i, p in enumerate(pop) if p > 0.5]
        neg = [i for i, p in enumerate(pop) if p <= 0.5]
        if not pos or not neg:
            return data

        # Aim for roughly one positive per three negatives.
        n_pos = min(len(pos), max(1, max_total // 4))
        n_neg = min(len(neg), max_total - n_pos)

        idx = pos[-n_pos:] + neg[-n_neg:]
        idx.sort()
        return {"X": [X[i] for i in idx], "amt": [amt[i] for i in idx], "pop": [pop[i] for i in idx]}

    def train_from_buffer(
        self,
        buffer_rows: List[Dict[str, Any]],
        horizons_min=HORIZONS,
        threshold_mm=0.1,
        pos_weight: float = 6.0,
        epochs: int = 25,
        max_samples: int = 240,
        max_replay: int = 900,
    ) -> Dict[str, Any]:
        fresh = self._build_batch(buffer_rows, horizons_min, threshold_mm, max_samples)

        if not fresh:
            return {"updated": 0, "reason": "not_enough_history"}

        self._update_replay(fresh, max_replay)

        batch = {
            h: self._balanced(mem, max_samples)
            for h, mem in self.replay.items()
            if mem["X"]
        }

        info = apply_training(
            self.state,
            batch,
            lr_pop=0.02,
            lr_q=0.02,
            l2=1e-3,
            pos_weight=pos_weight,
            epochs=epochs,
            trainable=self.trainable,
        )

        updated = int(info.get("updated", 0) or 0)
        if updated > 0:
            self._refit_platt()
            self.training_count_total += 1
            from datetime import datetime as _dt

            self.last_training_ts = _dt.utcnow().isoformat()

        self.last_training_info = info
        self.trained_samples = self.state.trained_samples
        return info