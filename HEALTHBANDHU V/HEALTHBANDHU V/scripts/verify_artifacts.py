"""Check that the models folder is complete, consistent and loadable.

Run from the project root:
    python scripts/verify_artifacts.py            # checks ./models
    python scripts/verify_artifacts.py path/to/models

Exit code 0 means the web app will start with these models.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from healthbandhu import config  # noqa: E402
from healthbandhu.models import ArtifactsMissingError, HealthBandhuPredictor, load_bundle  # noqa: E402

LINE = "=" * 64


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    models_dir = Path(argv[0]) if argv else config.MODELS_DIR

    print(LINE)
    print(f"HEALTHBANDHU ARTIFACT CHECK: {models_dir}")
    print(LINE)
    files = [
        (config.DNN_MODEL_FILE, "deep neural network"),
        (config.NB_PARAMS_FILE, "Naive Bayes parameters"),
        (config.SYMPTOMS_FILE, "symptom order (required)"),
        (config.DISEASES_FILE, "disease order (required)"),
        (config.PROFILES_FILE, "rule-layer profiles"),
        (config.METADATA_FILE, "metrics and settings"),
    ]
    for name, role in files:
        path = models_dir / name
        size = f"{path.stat().st_size / 1024**2:8.3f} MB" if path.exists() else "     missing"
        print(f"  {'OK ' if path.exists() else '-- '} {name:38s} {size}  {role}")

    print()
    try:
        bundle = load_bundle(models_dir)
    except ArtifactsMissingError as err:
        print(f"FAILED: {err}")
        return 1
    except Exception as err:  # shape mismatch, corrupt file, incompatible library version
        print(f"FAILED: {type(err).__name__}: {err}")
        return 1

    predictor = HealthBandhuPredictor(bundle)
    print(f"  Symptoms           : {len(bundle.symptoms)}")
    print(f"  Diseases           : {len(bundle.diseases)}")
    print(f"  Models in ensemble : {', '.join(predictor.models_used)}")
    print(f"  DNN weight         : {bundle.dnn_weight:.2f}")
    print(f"  NB temperature     : {bundle.nb_temperature:g}")
    print(f"  Rule cross-check   : {'available' if predictor.has_rules else 'unavailable'}")
    for note in bundle.warnings:
        print(f"  Note: {note}")

    frequency = bundle.metadata.get("symptom_train_frequency")
    if frequency and len(frequency) == len(bundle.symptoms):
        ranked = sorted(zip(frequency, bundle.symptoms), reverse=True)
        probe = [s for _, s in ranked[:4]]
    else:
        probe = bundle.symptoms[:4]

    result = predictor.diagnose(probe)
    total = float(predictor.ensemble_proba(predictor.vectorize(probe)[0][None, :]).sum())
    print()
    print(f"  Test symptoms      : {', '.join(probe)}")
    print(f"  Emergency level    : {result.emergency.level}")
    print(f"  Top condition      : {result.top.name} ({result.top.probability:.1%})")
    print(f"  Probabilities sum  : {total:.5f}")
    if abs(total - 1.0) > 1e-3:
        print("FAILED: probabilities do not sum to 1")
        return 1

    print()
    print("ALL CHECKS PASSED. Start the app with: streamlit run App.py")
    print(LINE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
