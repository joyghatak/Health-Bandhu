# HealthBandhu V

AI-assisted multiclass disease screening and clinical decision support, built by Joy Ghatak.

HealthBandhu takes a list of symptoms, screens them for red flags, ranks possible conditions with an ensemble of a deep neural network and a Bernoulli Naive Bayes model, explains which symptoms drove the result, cross-checks it with an independent rule engine and exports a PDF report.

> **Disclaimer.** HealthBandhu is an educational and research prototype. It does not provide a medical diagnosis and must not replace assessment by a qualified healthcare professional.

## How a symptom check runs

1. **Emergency screening:** weighted red-flag symptoms set the triage level (Critical, High risk, Moderate risk, Low risk).
2. **Deep neural network:** 512-256-128 network with batch normalisation and dropout.
3. **Bernoulli Naive Bayes:** classical probabilistic model with tuned smoothing and temperature.
4. **Ensemble:** weighted average of both models; the weight is tuned on validation data.
5. **Explainable AI:** occlusion attribution shows each symptom's share of the evidence.
6. **Rule-based cross-check:** key-symptom profiles for every disease, independent of machine learning.
7. **Confidence layer:** Very high, High, Moderate or Low, checked against real test accuracy.
8. **Clinical assistant and report:** plain-language next steps, PDF and JSON export.

## Model Performance & Evaluation

All figures are measured on **49,386 held-out test patients** (20% of the dataset) that were never used for training or validation tuning.

### Test Set Accuracy Comparison

| Model | Top-1 Accuracy | Top-3 Accuracy | Top-5 Accuracy | Macro F1 | Weighted F1 | In App |
|---|---|---|---|---|---|---|
| **Ensemble (Final)** | **87.28%** | **96.53%** | **98.32%** | **0.8491** | **0.8736** | Yes |
| **Bernoulli Naive Bayes** | **87.28%** | **96.53%** | **98.32%** | **0.8491** | **0.8736** | Yes |
| **Deep Neural Network (DNN)** | 85.97% | 95.79% | 97.91% | 0.8406 | 0.8611 | Yes |
| **Rule-based Cross-Check** | 81.16% | 94.10% | 96.91% | 0.7140 | 0.8331 | Yes |
| Extra Trees (Baseline) | 47.00% | 55.15% | 59.02% | 0.5855 | 0.5300 | No (Baseline only) |

> **Top-k accuracy:** The correct condition appears among the $k$ highest-ranked conditions. With Top-5 accuracy reaching **98.32%**, the system provides comprehensive differential screening across 754 conditions.

### Confidence Calibration

Real-world accuracy graded by model confidence levels across test patients:

| Confidence Level | Score Range | Test Patients | Share of Patients | Top Condition Correctness |
|---|---|---|---|---|
| **Very High** | $\ge 90\%$ | 37,317 | 75.56% | **99.95%** |
| **High** | $75\% - 90\%$ | 1,065 | 2.16% | **80.09%** |
| **Moderate** | $50\% - 75\%$ | 1,842 | 3.73% | **66.88%** |
| **Low** | $< 50\%$ | 9,162 | 18.55% | **49.33%** |

### Dataset Facts & Partitioning

- **Total records used**: 246,926 patient cases across **754 conditions** and **324 symptoms**.
- **Data split**:
  - **Training set**: 177,791 cases (72%)
  - **Validation set**: 19,749 cases (8%)
  - **Test set**: 49,386 cases (20%)
- **Data preparation**:
  - Merged duplicated symptom columns (`regurgitation.1` $\to$ `regurgitation`).
  - Removed 53 zero-activation symptom columns that never occurred in training cases.
  - Filtered 19 rare diseases with fewer than 2 records to enable stratified evaluation.

## Project structure

```
HEALTHBANDHU V/
├── App.py                          Streamlit web application (entry point)
├── requirements.txt                Packages for the web app
├── requirements-train.txt          Extra packages for retraining
├── run_app.bat / run_app.sh        One-click start on Windows / Linux / macOS
├── .streamlit/config.toml          Theme and server settings
├── healthbandhu/                   Shared package used by the app AND the notebook
│   ├── config.py                   Paths, emergency weights, thresholds (edit settings here)
│   ├── emergency.py                Phase 3: emergency detection layer
│   ├── rules.py                    Phase 4: rule-based clinical layer
│   ├── explain.py                  Phase 9: explainable AI (occlusion attribution)
│   ├── clinical.py                 Phases 10-11: confidence layer and clinical assistant
│   ├── report.py                   Phase 12: PDF report generator (ReportLab)
│   ├── models.py                   Phase 14: model loading and universal prediction pipeline
│   └── symptom_groups.py           Body-system grouping for browsing symptoms in the app
├── notebooks/
│   └── HealthBandhu_Training.ipynb Full training pipeline (Phases 0-15)
├── models/                         Trained model files (created by the notebook)
├── data/                           Dataset (downloaded automatically, not included)
├── reports/                        Plots, CSV metrics and a sample PDF (created by the notebook)
├── scripts/
│   ├── verify_artifacts.py         Checks the models folder is complete and loadable
│   └── colab_launch.py             Starts the app in Colab with a public link
├── tests/
│   └── test_pipeline.py            Unit tests (no dataset needed)
└── docs/
    └── FIXES.md                    Every problem fixed from the previous notebook
```

## 1. Train the models (Google Colab)

The trained models are not included, because they must be produced from the full dataset.

1. Open [Google Colab](https://colab.research.google.com), then File > Upload notebook and choose `notebooks/HealthBandhu_Training.ipynb`.
2. Optional: Runtime > Change runtime type > T4 GPU.
3. Runtime > Run all. When the first cell asks, upload `HEALTHBANDHU V.zip`.
4. The notebook downloads the dataset, trains and evaluates every layer, saves the models and finally downloads a new `HEALTHBANDHU V.zip` that **includes the trained models**.

## 2. Run the web app on your computer (Windows with Conda)

Extract the trained `HEALTHBANDHU V.zip`, then in Anaconda Prompt:

```bat
conda create -n healthbandhu python=3.11 -y
conda activate healthbandhu
cd "C:\path\to\HEALTHBANDHU V"
pip install -r requirements.txt
python scripts\verify_artifacts.py
streamlit run App.py
```

The app opens at http://localhost:8501. Next time, activate the environment and double-click `run_app.bat` (or run `streamlit run App.py`).

## 3. Public link straight from Colab

In the last notebook cell set `LAUNCH_APP_IN_COLAB = True` and run it. The launcher starts Streamlit, waits until it is healthy and prints a `trycloudflare.com` link that works while the Colab session stays connected.

## Deploy on Streamlit Community Cloud

1. Push the project, including the trained `models/` folder (a few MB), to a GitHub repository.
2. On [share.streamlit.io](https://share.streamlit.io) create an app with main file `App.py` and Python 3.11.

## Retrain locally instead of Colab

```bash
pip install -r requirements-train.txt
jupyter notebook notebooks/HealthBandhu_Training.ipynb
```

## Model files

| File | Content | Needed |
|---|---|---|
| `healthbandhu_dnn.keras` | Deep neural network | DNN or NB required |
| `healthbandhu_nb_params.npz` | Naive Bayes log probabilities, alpha, temperature | DNN or NB required |
| `healthbandhu_symptoms.json` | Symptom input order | Required |
| `healthbandhu_diseases.json` | Disease output order | Required |
| `healthbandhu_disease_profiles.npy` | Rule-layer profiles | Optional (rule check) |
| `model_metadata.json` | Measured metrics, ensemble weight, dataset facts, versions | Optional (performance page) |

No model is stored with pickle, so files load reliably across library versions. If a model is missing, the app keeps working with what is available and says so.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Troubleshooting

| Problem | Fix |
|---|---|
| App shows "No trained models found" | Run the notebook, then copy its `models/` folder into this project. |
| `UnpicklingError: STACK_GLOBAL requires str` (old version) | Caused by reading `joblib.dump` files with `pickle.load`. This version uses no pickle files. |
| Warning that the DNN was trained with a different TensorFlow | Install the version shown on the Model performance page. |
| `Port 8501 is already in use` | Run `streamlit run App.py --server.port 8502`. |
| Font looks plain | The Hind Siliguri font loads from Google Fonts; offline the system font is used. |

## Editing clinical settings

Emergency weights, triage thresholds, confidence levels and rule-layer settings are in `healthbandhu/config.py`. The app and the notebook both read them from there.
