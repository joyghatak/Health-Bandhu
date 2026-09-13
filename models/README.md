# Trained models

This folder is filled by `notebooks/HealthBandhu_Training.ipynb` (Phase 13). It is empty in the code-only release because the models must be trained on the full dataset.

Expected files:

- `healthbandhu_dnn.keras`: deep neural network
- `healthbandhu_nb_params.npz`: Bernoulli Naive Bayes parameters
- `healthbandhu_symptoms.json`: symptom input order (required)
- `healthbandhu_diseases.json`: disease output order (required)
- `healthbandhu_disease_profiles.npy`: rule-layer profiles
- `model_metadata.json`: measured metrics, ensemble weight, dataset facts, library versions

Check them with `python scripts/verify_artifacts.py`. To keep models elsewhere, set the `HEALTHBANDHU_MODELS_DIR` environment variable.
