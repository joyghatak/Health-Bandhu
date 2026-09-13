"""Phase 14: universal prediction pipeline.

Loads the trained artifacts and turns a list of symptoms into a complete,
explainable screening result. The notebook and the app both use this class.

Why no pickle for the classical model: the original notebook saved models with
joblib.dump and then read them with pickle.load, which fails with
"UnpicklingError: STACK_GLOBAL requires str" (joblib stores numpy arrays in its
own format). The Naive Bayes parameters are now plain numpy arrays (.npz), so
loading them needs no scikit-learn at all and cannot break on version changes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import config
from .clinical import ConfidenceAssessment, assess_confidence, build_recommendations
from .emergency import EmergencyAssessment, assess_emergency
from .explain import SymptomContribution, occlusion_attribution, unreported_typical_symptoms
from .rules import RuleMatch, key_symptom_mask, rule_scores, top_rule_matches

PathLike = Union[str, Path]


class ArtifactsMissingError(FileNotFoundError):
    """Raised when the models folder does not contain a usable model."""

    def __init__(self, models_dir: PathLike, missing: Sequence[str]):
        self.models_dir = Path(models_dir)
        self.missing = list(missing)
        super().__init__(f"Missing model artifacts in {self.models_dir}: {', '.join(self.missing)}")


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------
def _log_softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1, keepdims=True)
    return logits - (m + np.log(np.exp(logits - m).sum(axis=1, keepdims=True)))


def naive_bayes_proba(
    X: np.ndarray,
    feature_log_prob: np.ndarray,
    class_log_prior: np.ndarray,
    temperature: float = 1.0,
    dtype=np.float32,
) -> np.ndarray:
    """Bernoulli Naive Bayes posterior, identical to sklearn's BernoulliNB.predict_proba
    when temperature == 1. A temperature > 1 softens NB's typically over-confident
    probabilities (tuned on the validation set in the notebook)."""
    X = np.asarray(X, dtype=np.float64)
    flp = np.asarray(feature_log_prob, dtype=np.float64)
    neg = np.log1p(-np.exp(flp))
    joint = X @ (flp - neg).T + (np.asarray(class_log_prior, dtype=np.float64) + neg.sum(axis=1))
    return np.exp(_log_softmax(joint / float(temperature))).astype(dtype, copy=False)


def top_k_indices(proba: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest values per row, best first."""
    k = min(k, proba.shape[1])
    part = np.argpartition(-proba, k - 1, axis=1)[:, :k]
    rows = np.arange(proba.shape[0])[:, None]
    order = np.argsort(-proba[rows, part], axis=1, kind="stable")
    return part[rows, order]


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------
@dataclass
class DiseaseCandidate:
    rank: int
    name: str
    probability: float
    dnn_probability: Optional[float] = None
    nb_probability: Optional[float] = None
    rule_score: Optional[float] = None
    train_count: Optional[int] = None


@dataclass
class DiagnosisResult:
    symptoms: List[str]
    unknown_symptoms: List[str]
    emergency: EmergencyAssessment
    candidates: List[DiseaseCandidate]
    confidence: ConfidenceAssessment
    contributions: List[SymptomContribution]
    unreported_typical: List[Tuple[str, float]]
    rule_matches: List[RuleMatch]
    dnn_top: Optional[str]
    nb_top: Optional[str]
    recommendations: List[str]
    models_used: List[str]
    dnn_weight: float
    generated_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat(timespec="seconds"))

    @property
    def top(self) -> DiseaseCandidate:
        return self.candidates[0]

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "symptoms": self.symptoms,
            "unknown_symptoms": self.unknown_symptoms,
            "emergency": self.emergency.to_dict(),
            "confidence": self.confidence.to_dict(),
            "candidates": [c.__dict__ for c in self.candidates],
            "contributions": [c.to_dict() for c in self.contributions],
            "unreported_typical": [{"symptom": s, "frequency": p} for s, p in self.unreported_typical],
            "rule_matches": [m.to_dict() for m in self.rule_matches],
            "dnn_top": self.dnn_top,
            "nb_top": self.nb_top,
            "recommendations": self.recommendations,
            "models_used": self.models_used,
            "dnn_weight": self.dnn_weight,
            "disclaimer": config.DISCLAIMER,
        }


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------
@dataclass
class ModelBundle:
    symptoms: List[str]
    diseases: List[str]
    dnn: object = None
    nb_feature_log_prob: Optional[np.ndarray] = None
    nb_class_log_prior: Optional[np.ndarray] = None
    nb_temperature: float = 1.0
    profiles: Optional[np.ndarray] = None
    dnn_weight: float = config.DEFAULT_DNN_WEIGHT
    metadata: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.symptoms = [str(s) for s in self.symptoms]
        self.diseases = [str(d) for d in self.diseases]
        n_f, n_c = len(self.symptoms), len(self.diseases)
        if len(set(self.symptoms)) != n_f:
            raise ValueError("Symptom list contains duplicates.")
        if self.dnn is None and self.nb_feature_log_prob is None:
            raise ValueError("At least one model (DNN or Naive Bayes) is required.")
        if self.dnn is not None:
            in_dim = int(self.dnn.input_shape[-1])
            out_dim = int(self.dnn.output_shape[-1])
            if (in_dim, out_dim) != (n_f, n_c):
                raise ValueError(
                    f"DNN shape {in_dim}->{out_dim} does not match {n_f} symptoms and {n_c} diseases."
                )
        if self.nb_feature_log_prob is not None:
            if self.nb_feature_log_prob.shape != (n_c, n_f) or self.nb_class_log_prior.shape != (n_c,):
                raise ValueError("Naive Bayes parameter shapes do not match the symptom/disease lists.")
        if self.profiles is not None and self.profiles.shape != (n_c, n_f):
            raise ValueError("Disease profile matrix shape does not match the symptom/disease lists.")
        if self.dnn is None:
            self.dnn_weight = 0.0
        elif self.nb_feature_log_prob is None:
            self.dnn_weight = 1.0
        self.dnn_weight = float(min(max(self.dnn_weight, 0.0), 1.0))


def _load_name_list(folder: Path, json_name: str, legacy_name: str) -> Optional[List[str]]:
    if (folder / json_name).exists():
        with open(folder / json_name, "r", encoding="utf-8") as fh:
            return list(json.load(fh))
    if (folder / legacy_name).exists():
        import joblib  # legacy files were written with joblib.dump

        return list(joblib.load(folder / legacy_name))
    return None


def _major_minor(version: str) -> str:
    return ".".join(str(version).split(".")[:2])


def load_bundle(models_dir: Optional[PathLike] = None) -> ModelBundle:
    """Load every available artifact from the models folder."""
    folder = Path(models_dir) if models_dir is not None else config.MODELS_DIR
    warnings: List[str] = []

    symptoms = _load_name_list(folder, config.SYMPTOMS_FILE, config.LEGACY_SYMPTOMS_FILE)
    diseases = _load_name_list(folder, config.DISEASES_FILE, config.LEGACY_DISEASES_FILE)
    has_dnn = (folder / config.DNN_MODEL_FILE).exists()
    has_nb = (folder / config.NB_PARAMS_FILE).exists()

    missing = []
    if symptoms is None:
        missing.append(config.SYMPTOMS_FILE)
    if diseases is None:
        missing.append(config.DISEASES_FILE)
    if not (has_dnn or has_nb):
        missing.append(f"{config.DNN_MODEL_FILE} or {config.NB_PARAMS_FILE}")
    if missing:
        raise ArtifactsMissingError(folder, missing)

    metadata: Dict = {}
    if (folder / config.METADATA_FILE).exists():
        with open(folder / config.METADATA_FILE, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)
    else:
        warnings.append(f"{config.METADATA_FILE} not found: performance figures and tuned weights are unavailable.")

    dnn = None
    if has_dnn:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        trained = metadata.get("versions", {})
        try:
            import keras

            dnn = keras.models.load_model(folder / config.DNN_MODEL_FILE, compile=False)
            installed_keras = getattr(keras, "__version__", None)
            if installed_keras and trained.get("keras") and _major_minor(trained["keras"]) != _major_minor(installed_keras):
                warnings.append(
                    f"The DNN was trained with keras {trained['keras']} but {installed_keras} is installed."
                )
        except Exception:
            try:
                import tensorflow as tf

                dnn = tf.keras.models.load_model(folder / config.DNN_MODEL_FILE, compile=False)
                for lib, installed in (("tensorflow", tf.__version__), ("keras", getattr(tf.keras, "__version__", None))):
                    if installed and trained.get(lib) and _major_minor(trained[lib]) != _major_minor(installed):
                        warnings.append(
                            f"The DNN was trained with {lib} {trained[lib]} but {installed} is installed."
                        )
            except Exception as err:
                warnings.append(f"Deep learning model could not be loaded: {err}. Running with Naive Bayes only.")

    nb_flp = nb_clp = None
    nb_temperature = 1.0
    if has_nb:
        with np.load(folder / config.NB_PARAMS_FILE) as npz:
            nb_flp = npz["feature_log_prob"].astype(np.float64)
            nb_clp = npz["class_log_prior"].astype(np.float64)
            if "temperature" in npz:
                nb_temperature = float(npz["temperature"])
    elif has_dnn:
        warnings.append("Naive Bayes parameters not found: running with the deep learning model only.")

    profiles = None
    if (folder / config.PROFILES_FILE).exists():
        profiles = np.load(folder / config.PROFILES_FILE).astype(np.float32)
    else:
        warnings.append("Disease profiles not found: the rule-based cross-check is disabled.")

    dnn_weight = metadata.get("ensemble", {}).get("dnn_weight", config.DEFAULT_DNN_WEIGHT)

    return ModelBundle(
        symptoms=symptoms,
        diseases=diseases,
        dnn=dnn,
        nb_feature_log_prob=nb_flp,
        nb_class_log_prior=nb_clp,
        nb_temperature=nb_temperature,
        profiles=profiles,
        dnn_weight=dnn_weight,
        metadata=metadata,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------
class HealthBandhuPredictor:
    """Emergency screening -> ensemble prediction -> explanation -> recommendations."""

    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle
        self.symptoms = bundle.symptoms
        self.diseases = bundle.diseases
        self._index = {s: i for i, s in enumerate(self.symptoms)}
        self._key_mask = key_symptom_mask(bundle.profiles) if bundle.profiles is not None else None
        counts = bundle.metadata.get("class_train_counts")
        self._class_counts = list(counts) if counts and len(counts) == len(self.diseases) else None
        self._dnn_head = self._build_precise_head(bundle.dnn)

    @staticmethod
    def _build_precise_head(dnn):
        """Split the DNN into (feature extractor, final kernel, final bias) so the last
        softmax can run in float64. float32 softmax rounds confident outputs to exactly
        1.0, which would make every occlusion explanation zero. Returns None if the
        network does not end in a softmax Dense layer."""
        if dnn is None:
            return None
        try:
            last = dnn.layers[-1]
            activation = getattr(getattr(last, "activation", None), "__name__", "")
            if activation != "softmax":
                return None
            try:
                import keras

                features = keras.Model(dnn.inputs[0], dnn.layers[-2].output)
            except Exception:
                import tensorflow as tf

                features = tf.keras.Model(dnn.inputs[0], dnn.layers[-2].output)
            kernel, bias = last.get_weights()
            return features, kernel.astype(np.float64), bias.astype(np.float64)
        except Exception:
            return None

    # -- capabilities ------------------------------------------------------
    @property
    def has_dnn(self) -> bool:
        return self.bundle.dnn is not None

    @property
    def has_nb(self) -> bool:
        return self.bundle.nb_feature_log_prob is not None

    @property
    def has_rules(self) -> bool:
        return self._key_mask is not None

    @property
    def models_used(self) -> List[str]:
        used = []
        if self.has_dnn and self.bundle.dnn_weight > 0:
            used.append("Deep neural network")
        if self.has_nb and self.bundle.dnn_weight < 1:
            used.append("Bernoulli Naive Bayes")
        return used

    # -- vectors -------------------------------------------------------------
    def vectorize(self, symptoms: Iterable[str]) -> Tuple[np.ndarray, List[str], List[str]]:
        x = np.zeros(len(self.symptoms), dtype=np.float32)
        known, unknown = [], []
        for raw in symptoms:
            s = str(raw).strip().lower()
            if s in self._index:
                if x[self._index[s]] == 0:
                    known.append(s)
                x[self._index[s]] = 1.0
            elif s:
                unknown.append(s)
        return x, known, unknown

    # -- probabilities ---------------------------------------------------------
    def dnn_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.shape[0] <= 2048:
            if self._dnn_head is not None:
                features, kernel, bias = self._dnn_head
                hidden = np.asarray(features(X, training=False), dtype=np.float64)
                return np.exp(_log_softmax(hidden @ kernel + bias))
            return np.asarray(self.bundle.dnn(X, training=False), dtype=np.float64)
        return self.bundle.dnn.predict(X, batch_size=4096, verbose=0).astype(np.float32)

    def nb_proba(self, X: np.ndarray, batch_size: int = 8192, dtype=np.float32) -> np.ndarray:
        X = np.asarray(X)
        parts = [
            naive_bayes_proba(
                X[i:i + batch_size],
                self.bundle.nb_feature_log_prob,
                self.bundle.nb_class_log_prior,
                self.bundle.nb_temperature,
                dtype=dtype,
            )
            for i in range(0, X.shape[0], batch_size)
        ]
        return np.concatenate(parts, axis=0)

    def predict_components(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Per-model and ensemble probabilities. Small batches (interactive use) are
        combined in float64 so explanations keep precision near 0% and 100%."""
        out: Dict[str, np.ndarray] = {}
        w = self.bundle.dnn_weight
        dtype = np.float64 if np.shape(X)[0] <= 2048 else np.float32
        if self.has_dnn:
            out["dnn"] = self.dnn_proba(X).astype(dtype, copy=False)
        if self.has_nb:
            out["nb"] = self.nb_proba(X, dtype=dtype)
        if "dnn" in out and "nb" in out:
            out["ensemble"] = w * out["dnn"] + (1.0 - w) * out["nb"]
        else:
            out["ensemble"] = out.get("dnn", out.get("nb"))
        return out

    def ensemble_proba(self, X: np.ndarray) -> np.ndarray:
        return self.predict_components(X)["ensemble"]

    # -- full pipeline -------------------------------------------------------
    def diagnose(self, symptoms: Iterable[str], top_k: int = config.TOP_K) -> DiagnosisResult:
        x, known, unknown = self.vectorize(symptoms)
        if not known:
            raise ValueError("Select at least one recognised symptom.")

        emergency = assess_emergency(known)
        comps = self.predict_components(x[None, :])
        p = comps["ensemble"][0]
        top = top_k_indices(p[None, :], top_k)[0]

        rule_row = None
        rule_matches: List[RuleMatch] = []
        if self.has_rules:
            rule_row = rule_scores(x[None, :], self.bundle.profiles, self._key_mask)[0]
            rule_matches = top_rule_matches(
                x, self.symptoms, self.diseases, self.bundle.profiles, self._key_mask, top_k
            )

        candidates = [
            DiseaseCandidate(
                rank=r + 1,
                name=self.diseases[i],
                probability=float(p[i]),
                dnn_probability=float(comps["dnn"][0, i]) if "dnn" in comps else None,
                nb_probability=float(comps["nb"][0, i]) if "nb" in comps else None,
                rule_score=float(rule_row[i]) if rule_row is not None else None,
                train_count=int(self._class_counts[i]) if self._class_counts else None,
            )
            for r, i in enumerate(top)
        ]

        second = float(p[top[1]]) if len(top) > 1 else 0.0
        confidence = assess_confidence(float(p[top[0]]), second)
        _, contributions = occlusion_attribution(self.ensemble_proba, x, int(top[0]), self.symptoms)

        unreported = []
        if self.bundle.profiles is not None:
            unreported = unreported_typical_symptoms(self.bundle.profiles[top[0]], self.symptoms, set(known))

        dnn_top = self.diseases[int(np.argmax(comps["dnn"][0]))] if "dnn" in comps else None
        nb_top = self.diseases[int(np.argmax(comps["nb"][0]))] if "nb" in comps else None
        both = self.has_dnn and self.has_nb and 0 < self.bundle.dnn_weight < 1

        recommendations = build_recommendations(
            emergency=emergency,
            confidence=confidence,
            n_symptoms=len(known),
            top_condition=candidates[0].name,
            top_condition_train_count=candidates[0].train_count,
            dnn_top=dnn_top if both else None,
            nb_top=nb_top if both else None,
        )

        return DiagnosisResult(
            symptoms=known,
            unknown_symptoms=unknown,
            emergency=emergency,
            candidates=candidates,
            confidence=confidence,
            contributions=contributions,
            unreported_typical=unreported,
            rule_matches=rule_matches,
            dnn_top=dnn_top,
            nb_top=nb_top,
            recommendations=recommendations,
            models_used=self.models_used,
            dnn_weight=self.bundle.dnn_weight,
        )
