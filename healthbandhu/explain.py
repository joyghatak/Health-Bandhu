"""Phase 9: explainable AI layer.

The original notebook's "explanation" just listed the symptoms the patient
reported. Here each reported symptom is removed one at a time and the drop in
the predicted condition's probability is measured (occlusion attribution). The
explanation therefore reflects what the ensemble actually relied on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, List, Sequence, Set, Tuple

import numpy as np

from . import config

ProbaFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class SymptomContribution:
    symptom: str
    impact: float            # probability (0-1) the condition loses when this symptom is removed
    log_impact: float = 0.0  # drop in log-probability; stays informative when probability is near 100%
    influence: float = 0.0   # log_impact / sum(|log_impact|): signed share of the total evidence

    def to_dict(self) -> dict:
        return asdict(self)


EPS = 1e-12


def occlusion_attribution(
    proba_fn: ProbaFn,
    x: np.ndarray,
    target_index: int,
    symptoms: Sequence[str],
) -> Tuple[float, List[SymptomContribution]]:
    """Return (probability with all symptoms, contributions sorted by impact)."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    active = np.flatnonzero(x > 0)
    batch = np.repeat(x[None, :], active.size + 1, axis=0)
    if active.size:
        batch[np.arange(1, active.size + 1), active] = 0.0
    probs = np.asarray(proba_fn(batch), dtype=np.float64)[:, target_index]
    base = float(probs[0])
    log_probs = np.log(np.maximum(probs, EPS))
    log_impacts = log_probs[0] - log_probs[1:]
    total = float(np.abs(log_impacts).sum())
    contributions = [
        SymptomContribution(
            symptom=str(symptoms[a]),
            impact=float(base - probs[i + 1]),
            log_impact=float(log_impacts[i]),
            influence=float(log_impacts[i] / total) if total > 0 else 0.0,
        )
        for i, a in enumerate(active)
    ]
    contributions.sort(key=lambda c: (c.log_impact, c.impact), reverse=True)
    return base, contributions


def unreported_typical_symptoms(
    profile_row: np.ndarray,
    symptoms: Sequence[str],
    reported: Set[str],
    min_prob: float = config.TYPICAL_SYMPTOM_MIN_PROB,
    limit: int = 6,
) -> List[Tuple[str, float]]:
    """Symptoms common in the predicted condition that the patient did not report."""
    order = np.argsort(-np.asarray(profile_row), kind="stable")
    out = []
    for i in order:
        p = float(profile_row[i])
        if p < min_prob or len(out) >= limit:
            break
        if symptoms[i] not in reported:
            out.append((str(symptoms[i]), p))
    return out


def global_occlusion_importance(
    proba_fn: ProbaFn,
    X_sample: np.ndarray,
    symptoms: Sequence[str],
) -> List[Tuple[str, float, int]]:
    """Average occlusion impact of each symptom over a sample of patients.

    Returns [(symptom, mean influence share on the predicted class, times active)], sorted.
    """
    X_sample = np.asarray(X_sample, dtype=np.float32)
    position = {s: j for j, s in enumerate(symptoms)}
    totals = np.zeros(X_sample.shape[1], dtype=np.float64)
    counts = np.zeros(X_sample.shape[1], dtype=np.int64)
    for x in X_sample:
        if not np.any(x > 0):
            continue
        target = int(np.argmax(proba_fn(x[None, :])[0]))
        _, contribs = occlusion_attribution(proba_fn, x, target, symptoms)
        for c in contribs:
            j = position[c.symptom]
            totals[j] += c.influence
            counts[j] += 1
    means = np.divide(totals, np.maximum(counts, 1))
    ranked = sorted(
        ((str(symptoms[j]), float(means[j]), int(counts[j])) for j in np.flatnonzero(counts)),
        key=lambda t: t[1],
        reverse=True,
    )
    return ranked
