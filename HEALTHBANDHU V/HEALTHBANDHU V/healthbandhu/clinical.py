"""Phases 10-11: confidence layer and clinical assistant.

Bug fixed from the original notebook: clinical_assistant() was called with the
confidence level ("VERY HIGH") in the emergency-level argument, so the report
printed "Emergency Level : VERY HIGH". Emergency and confidence are now
separate typed objects and cannot be swapped.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional

from . import config
from .emergency import EmergencyAssessment


@dataclass
class ConfidenceAssessment:
    level: str
    probability: float
    margin: float      # top-1 minus top-2 probability
    ambiguous: bool

    def to_dict(self) -> dict:
        return asdict(self)


def confidence_level(probability: float) -> str:
    for level, threshold in config.CONFIDENCE_LEVELS:
        if probability >= threshold:
            return level
    return config.LOW_CONFIDENCE


def assess_confidence(top1_probability: float, top2_probability: float = 0.0) -> ConfidenceAssessment:
    margin = float(top1_probability - top2_probability)
    return ConfidenceAssessment(
        level=confidence_level(top1_probability),
        probability=float(top1_probability),
        margin=margin,
        ambiguous=margin < config.AMBIGUITY_MARGIN,
    )


def build_recommendations(
    emergency: EmergencyAssessment,
    confidence: ConfidenceAssessment,
    n_symptoms: int,
    top_condition: str,
    top_condition_train_count: Optional[int] = None,
    dnn_top: Optional[str] = None,
    nb_top: Optional[str] = None,
) -> List[str]:
    """Plain-language, safety-first guidance for the result screen and PDF."""
    recs = [emergency.action]
    if emergency.is_urgent:
        recs.append(config.EMERGENCY_CONTACT_NOTE)

    if confidence.level in ("VERY HIGH", "HIGH"):
        recs.append(
            f"The models point clearly to {top_condition}, but this is a screening result, "
            "not a diagnosis. Confirm it with a qualified clinician."
        )
    elif confidence.level == "MODERATE":
        recs.append(
            "Several conditions fit these symptoms. Review the full list of possible "
            "conditions with a clinician."
        )
    else:
        recs.append(
            "The symptom pattern is not specific enough for a confident result. Add any other "
            "symptoms present, or seek a clinical assessment."
        )

    if confidence.ambiguous and confidence.level != config.LOW_CONFIDENCE:
        recs.append("The top two conditions have similar probabilities, so treat the ranking with caution.")

    if n_symptoms < config.MIN_SYMPTOMS_FOR_RELIABLE_RESULT:
        recs.append(
            f"Only {n_symptoms} symptom{'s' if n_symptoms != 1 else ''} reported. Results based on "
            "very few symptoms are unreliable."
        )

    if top_condition_train_count is not None and top_condition_train_count < config.RARE_CLASS_TRAIN_COUNT:
        recs.append(
            f"{top_condition} had only {top_condition_train_count} training examples, so its "
            "probability is less reliable than for common conditions."
        )

    if dnn_top and nb_top and dnn_top != nb_top:
        recs.append(
            f"The deep learning model favours {dnn_top} while the Naive Bayes model favours "
            f"{nb_top}. Model disagreement lowers reliability."
        )

    recs.append("Do not start, stop or change any medicine based on this result.")
    return recs
