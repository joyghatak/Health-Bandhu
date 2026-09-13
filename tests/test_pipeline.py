"""Smoke tests for the HealthBandhu package. No dataset or trained models needed.

Run from the project root:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from healthbandhu import config  # noqa: E402
from healthbandhu.clinical import assess_confidence, build_recommendations  # noqa: E402
from healthbandhu.emergency import assess_emergency, classify_emergency, emergency_scores  # noqa: E402
from healthbandhu.explain import occlusion_attribution  # noqa: E402
from healthbandhu.models import (  # noqa: E402
    ArtifactsMissingError, HealthBandhuPredictor, ModelBundle, load_bundle, naive_bayes_proba, top_k_indices,
)
from healthbandhu.rules import build_disease_profiles, key_symptom_mask, top_rule_matches  # noqa: E402

try:
    import sklearn  # noqa: F401
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False


SYMPTOMS = ["sharp chest pain", "shortness of breath", "cough", "fever", "sore throat",
            "headache", "nausea", "vomiting", "skin rash", "back pain"]
DISEASES = ["condition a", "condition b", "condition c"]


def synthetic_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    probs = np.full((3, len(SYMPTOMS)), 0.03)
    probs[0, [0, 1]] = 0.9          # chest pain + breathlessness
    probs[1, [2, 3, 4]] = 0.85      # cough, fever, sore throat
    probs[2, [6, 7, 5]] = 0.8       # nausea, vomiting, headache
    y = rng.integers(0, 3, size=n)
    X = (rng.random((n, len(SYMPTOMS))) < probs[y]).astype(np.uint8)
    return X, y


def fit_nb_params(X, y, alpha=1.0):
    counts = np.bincount(y, minlength=3)
    feat = np.stack([X[y == c].sum(axis=0) for c in range(3)])
    flp = np.log((feat + alpha) / (counts[:, None] + 2 * alpha))
    clp = np.log(counts / counts.sum())
    return flp, clp


class EmergencyTests(unittest.TestCase):
    def test_levels(self):
        self.assertEqual(classify_emergency(10), "CRITICAL")
        self.assertEqual(classify_emergency(5), "HIGH RISK")
        self.assertEqual(classify_emergency(2), "MODERATE RISK")
        self.assertEqual(classify_emergency(0), "LOW RISK")

    def test_assessment(self):
        a = assess_emergency(["Sharp chest pain ", "shortness of breath", "cough"])
        self.assertEqual((a.score, a.level), (10, "CRITICAL"))
        self.assertEqual([t[0] for t in a.triggers], ["sharp chest pain", "shortness of breath"])
        self.assertEqual(assess_emergency([]).level, "LOW RISK")

    def test_vectorised_matches_single(self):
        X, _ = synthetic_data(50)
        vec = emergency_scores(X, SYMPTOMS)
        for row, score in zip(X, vec):
            active = [SYMPTOMS[i] for i in np.flatnonzero(row)]
            self.assertEqual(assess_emergency(active).score, score)


class ClinicalTests(unittest.TestCase):
    def test_emergency_and_confidence_not_swapped(self):
        em = assess_emergency(["cough"])
        conf = assess_confidence(0.95, 0.02)
        recs = build_recommendations(em, conf, 1, "condition b")
        self.assertEqual(conf.level, "VERY HIGH")
        self.assertEqual(recs[0], config.EMERGENCY_ACTIONS["LOW RISK"])
        self.assertTrue(any("Only 1 symptom reported" in r for r in recs))


class NumericsTests(unittest.TestCase):
    @unittest.skipUnless(HAS_SKLEARN, "scikit-learn not installed")
    def test_naive_bayes_matches_sklearn(self):
        from sklearn.naive_bayes import BernoulliNB

        X, y = synthetic_data()
        nb = BernoulliNB(alpha=0.5).fit(X, y)
        ours = naive_bayes_proba(X, nb.feature_log_prob_, nb.class_log_prior_)
        np.testing.assert_allclose(ours, nb.predict_proba(X), atol=1e-5)

    def test_temperature_softens(self):
        X, y = synthetic_data()
        flp, clp = fit_nb_params(X, y)
        sharp = naive_bayes_proba(X, flp, clp, 1.0).max(axis=1).mean()
        soft = naive_bayes_proba(X, flp, clp, 5.0).max(axis=1).mean()
        self.assertLess(soft, sharp)

    def test_top_k(self):
        p = np.array([[0.1, 0.5, 0.2, 0.2], [0.7, 0.1, 0.15, 0.05]])
        np.testing.assert_array_equal(top_k_indices(p, 2), [[1, 2], [0, 2]])


class RuleAndExplainTests(unittest.TestCase):
    def test_rules(self):
        X, y = synthetic_data()
        profiles, counts = build_disease_profiles(X, y, 3)
        self.assertEqual(profiles.shape, (3, len(SYMPTOMS)))
        self.assertEqual(counts.sum(), len(y))
        mask = key_symptom_mask(profiles)
        x = np.zeros(len(SYMPTOMS), dtype=np.float32)
        x[[2, 3, 4]] = 1
        best = top_rule_matches(x, SYMPTOMS, DISEASES, profiles, mask, top_k=3)[0]
        self.assertEqual(best.disease, "condition b")
        self.assertEqual(set(best.matched), {"cough", "fever", "sore throat"})

    def test_occlusion(self):
        w = np.array([[2.0, 1.0, 0.0]])

        def proba_fn(batch):
            logits = np.concatenate([batch @ w.T, np.zeros((len(batch), 1))], axis=1)
            e = np.exp(logits)
            return e / e.sum(axis=1, keepdims=True)

        base, contribs = occlusion_attribution(proba_fn, np.array([1, 1, 1]), 0, ["a", "b", "c"])
        self.assertGreater(base, 0.9)
        self.assertEqual([c.symptom for c in contribs], ["a", "b", "c"])
        self.assertAlmostEqual(contribs[-1].impact, 0.0, places=6)


class PredictorTests(unittest.TestCase):
    def make_bundle(self, with_dnn):
        X, y = synthetic_data()
        flp, clp = fit_nb_params(X, y)
        profiles, counts = build_disease_profiles(X, y, 3)
        dnn = None
        if with_dnn:
            tf.keras.utils.set_random_seed(0)
            dnn = tf.keras.Sequential([tf.keras.Input(shape=(len(SYMPTOMS),)),
                                       tf.keras.layers.Dense(16, activation="relu"),
                                       tf.keras.layers.Dense(3, activation="softmax")])
            dnn.compile("adam", "sparse_categorical_crossentropy")
            dnn.fit(X.astype("float32"), y, epochs=30, verbose=0)
        return ModelBundle(symptoms=SYMPTOMS, diseases=DISEASES, dnn=dnn, nb_feature_log_prob=flp,
                           nb_class_log_prior=clp, profiles=profiles, dnn_weight=0.7,
                           metadata={"class_train_counts": counts.tolist()})

    def check_result(self, predictor):
        result = predictor.diagnose(["sharp chest pain", "shortness of breath", "not a symptom"])
        self.assertEqual(result.unknown_symptoms, ["not a symptom"])
        self.assertEqual(result.emergency.level, "CRITICAL")
        self.assertEqual(result.top.name, "condition a")
        probs = [c.probability for c in result.candidates]
        self.assertEqual(probs, sorted(probs, reverse=True))
        full = predictor.ensemble_proba(predictor.vectorize(result.symptoms)[0][None, :])
        self.assertAlmostEqual(float(full.sum()), 1.0, places=4)
        self.assertEqual(result.recommendations[0], config.EMERGENCY_ACTIONS["CRITICAL"])
        from healthbandhu.report import build_pdf_report

        pdf = build_pdf_report(result, {"name": "Test <&> Patient", "age": 40, "sex": "Female"})
        self.assertTrue(pdf.startswith(b"%PDF"))
        with self.assertRaises(ValueError):
            predictor.diagnose(["not a symptom"])

    def test_naive_bayes_only(self):
        bundle = self.make_bundle(with_dnn=False)
        self.assertEqual(bundle.dnn_weight, 0.0)
        self.check_result(HealthBandhuPredictor(bundle))

    @unittest.skipUnless(HAS_TF, "TensorFlow not installed")
    def test_ensemble(self):
        predictor = HealthBandhuPredictor(self.make_bundle(with_dnn=True))
        self.assertEqual(len(predictor.models_used), 2)
        self.check_result(predictor)

    def test_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ArtifactsMissingError):
                load_bundle(tmp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
