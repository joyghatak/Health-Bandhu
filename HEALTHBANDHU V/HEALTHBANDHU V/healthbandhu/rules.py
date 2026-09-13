"""Phase 4: rule-based clinical layer.

Each disease gets a symptom profile: the fraction of its training patients who
reported each symptom. Symptoms seen in at least KEY_SYMPTOM_THRESHOLD of those
patients become the disease's "key symptoms". A patient is matched against every
disease (the original notebook only covered the 20 most frequent diseases and
built profiles from the full dataset, leaking test data).

Rule score = RULE_F1_WEIGHT * F1(reported symptoms, key symptoms)
           + (1 - RULE_F1_WEIGHT) * mean profile frequency of the reported symptoms
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

from . import config


@dataclass
class RuleMatch:
    disease: str
    score: float
    matched: List[str] = field(default_factory=list)   # reported AND key symptom
    missing: List[str] = field(default_factory=list)   # key symptom NOT reported
    key_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def build_disease_profiles(X, y, n_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (profiles [n_classes x n_symptoms] float32, class counts [n_classes])."""
    X = np.asarray(X)
    y = np.asarray(y)
    counts = np.bincount(y, minlength=n_classes)
    order = np.argsort(y, kind="stable")
    bounds = np.searchsorted(y[order], np.arange(n_classes + 1))
    profiles = np.zeros((n_classes, X.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        idx = order[bounds[c]:bounds[c + 1]]
        if idx.size:
            profiles[c] = X[idx].mean(axis=0, dtype=np.float64)
    return profiles, counts


def key_symptom_mask(
    profiles: np.ndarray,
    threshold: float = config.KEY_SYMPTOM_THRESHOLD,
    min_k: int = config.KEY_SYMPTOMS_MIN,
    max_k: int = config.KEY_SYMPTOMS_MAX,
) -> np.ndarray:
    """Boolean mask of key symptoms per disease."""
    profiles = np.asarray(profiles, dtype=np.float32)
    n_classes, n_features = profiles.shape
    order = np.argsort(-profiles, axis=1, kind="stable")
    ranks = np.empty_like(order)
    ranks[np.arange(n_classes)[:, None], order] = np.arange(n_features)[None, :]
    mask = (profiles >= threshold) & (ranks < max_k)
    mask |= (ranks < min_k) & (profiles > 0)
    return mask


def rule_scores(
    X,
    profiles: np.ndarray,
    key_mask: np.ndarray,
    f1_weight: float = config.RULE_F1_WEIGHT,
    batch_size: int = 4096,
) -> np.ndarray:
    """Rule score for every (patient, disease) pair, in [0, 1]."""
    X = np.asarray(X)
    K = key_mask.astype(np.float32)
    P = np.asarray(profiles, dtype=np.float32)
    key_counts = K.sum(axis=1)[None, :]
    n = X.shape[0]
    out = np.empty((n, P.shape[0]), dtype=np.float32)
    for start in range(0, n, batch_size):
        xb = X[start:start + batch_size].astype(np.float32)
        reported = xb.sum(axis=1, keepdims=True)
        inter = xb @ K.T
        f1 = 2.0 * inter / np.maximum(reported + key_counts, 1.0)
        coverage = (xb @ P.T) / np.maximum(reported, 1.0)
        out[start:start + xb.shape[0]] = f1_weight * f1 + (1.0 - f1_weight) * coverage
    return out


def top_rule_matches(
    x: np.ndarray,
    symptoms: Sequence[str],
    diseases: Sequence[str],
    profiles: np.ndarray,
    key_mask: np.ndarray,
    top_k: int = config.TOP_K,
) -> List[RuleMatch]:
    """Explainable rule matches for a single patient vector."""
    x = np.asarray(x).reshape(1, -1)
    scores = rule_scores(x, profiles, key_mask)[0]
    reported = x[0] > 0
    best = np.argsort(-scores, kind="stable")[:top_k]
    matches = []
    for d in best:
        key = key_mask[d]
        matched_idx = np.flatnonzero(key & reported)
        missing_idx = np.flatnonzero(key & ~reported)
        missing_idx = missing_idx[np.argsort(-profiles[d, missing_idx], kind="stable")]
        matches.append(
            RuleMatch(
                disease=str(diseases[d]),
                score=float(scores[d]),
                matched=[str(symptoms[i]) for i in matched_idx],
                missing=[str(symptoms[i]) for i in missing_idx],
                key_count=int(key.sum()),
            )
        )
    return matches
