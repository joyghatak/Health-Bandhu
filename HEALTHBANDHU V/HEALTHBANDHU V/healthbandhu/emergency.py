"""Phase 3: emergency detection layer.

Screens reported symptoms for red flags before any disease prediction is shown.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from . import config


@dataclass
class EmergencyAssessment:
    score: int
    level: str
    action: str
    triggers: List[List] = field(default_factory=list)  # [[symptom, weight], ...]

    @property
    def is_urgent(self) -> bool:
        return self.level in ("CRITICAL", "HIGH RISK")

    def to_dict(self) -> dict:
        return asdict(self)


def classify_emergency(score: float) -> str:
    """Map a numeric emergency score to a triage level."""
    for level, threshold in config.EMERGENCY_LEVELS:
        if score >= threshold:
            return level
    return config.LOW_RISK


def assess_emergency(
    active_symptoms: Iterable[str],
    weights: Optional[Dict[str, int]] = None,
) -> EmergencyAssessment:
    """Score one patient's reported symptoms."""
    weights = config.EMERGENCY_SYMPTOMS if weights is None else weights
    active = {str(s).strip().lower() for s in active_symptoms}
    triggers = sorted(
        ((symptom, int(w)) for symptom, w in weights.items() if symptom in active),
        key=lambda item: (-item[1], item[0]),
    )
    score = int(sum(w for _, w in triggers))
    level = classify_emergency(score)
    return EmergencyAssessment(
        score=score,
        level=level,
        action=config.EMERGENCY_ACTIONS[level],
        triggers=[[s, w] for s, w in triggers],
    )


def emergency_scores(
    X,
    symptoms: Sequence[str],
    weights: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    """Vectorised emergency score for every row of a binary symptom matrix.

    Only the emergency columns are touched, so this is fast even for the full
    246k-row dataset (the original notebook used a slow row-by-row apply).
    """
    weights = config.EMERGENCY_SYMPTOMS if weights is None else weights
    index = {s: i for i, s in enumerate(symptoms)}
    cols = [index[s] for s in weights if s in index]
    if not cols:
        return np.zeros(len(X), dtype=np.int32)
    w = np.array([weights[symptoms[c]] for c in cols], dtype=np.int32)
    X = np.asarray(X)
    return (X[:, cols].astype(np.int32) @ w).astype(np.int32)


def emergency_symptoms_in(symptoms: Sequence[str], weights: Optional[Dict[str, int]] = None) -> List[str]:
    """Emergency symptoms that actually exist in the model's symptom list."""
    weights = config.EMERGENCY_SYMPTOMS if weights is None else weights
    present = set(symptoms)
    return [s for s in weights if s in present]
