# Problems fixed from `HEALTH__BANDHU_Final (2).ipynb`

Each entry names the original cell or phase, what was wrong, and what HealthBandhu V does instead.

## Application and deployment

| # | Where | Problem | Fix |
|---|---|---|---|
| 1 | Audit cell, launcher cell | `app.py` was never written. The audit reported it missing, yet the launcher still ran `streamlit run app.py`, so the public link served an error. | `App.py` is a complete Streamlit app: symptom check, model performance and about pages, setup screen when models are missing. |
| 2 | GitHub release zip cell | The zip was 0.02 MB: it skipped the DNN model and listed `app.py`, `requirements.txt`, `README.md` and a notebook name that did not exist. | Phase 15 zips the whole project with trained models (dataset excluded). All listed files exist. |
| 3 | One-click launcher | `time.sleep(8)` race with Streamlit start-up, a blocking `readline()` loop, cloudflared downloaded twice (`/usr/local/bin` and `./cloudflared`). | `scripts/colab_launch.py` waits for Streamlit's health endpoint, uses time-outs, downloads cloudflared once and logs to `tools/`. |
| 4 | Zip cell | `google.colab.files` was called unconditionally and crashes outside Colab. | Colab-only calls are guarded by `IN_COLAB`. |

## Model files and loading

| # | Where | Problem | Fix |
|---|---|---|---|
| 5 | Audit and "deep fix" cells | `UnpicklingError: STACK_GLOBAL requires str` was blamed on a scikit-learn version mismatch, although both versions were 1.6.1. The real cause: files saved with `joblib.dump` were read with `pickle.load`. The latin-1 and identity `FixedUnpickler` attempts could not work, and the re-save workaround created 880 MB backup copies. | No pickle files. Symptom and disease lists are JSON, Naive Bayes parameters are `.npz`, the DNN is `.keras`. Legacy `.pkl` lists are still readable through joblib. `scripts/verify_artifacts.py` replaces the audit cells. |
| 6 | Phase 5, model saving | Extra Trees reached 52.8% accuracy and its file was 880 MB, because every tree node stores one value per class (754). It could not go on GitHub or Streamlit Cloud. | The app uses Bernoulli Naive Bayes (two small arrays). Extra Trees stays as a comparison baseline and is never saved; the notebook prints its would-be size. |
| 7 | Metrics cell | Metrics were typed in by hand (`top1 0.8570`, `top3 0.9587`, `top5 0.9792`) and did not match the computed results (0.8602, 0.9590, 0.9795). `records 246945` was the count before filtering. | `model_metadata.json` stores only computed values: top-1/3/5, macro and weighted F1 for every model, plus dataset facts and library versions. |
| 8 | Phases 6 and 15 | Hard-coded `input_shape=(377,)` (Keras warning) and `np.arange(754)`, with `all_classes` redefined after it was used. | Shapes come from the data through an explicit `Input` layer. |

## Data preparation

| # | Where | Problem | Fix |
|---|---|---|---|
| 9 | Dataset loader | The CSV repeats the `regurgitation` column; pandas renamed the copy `regurgitation.1`, so the same symptom was two features and the app could only set one of them. | Duplicated columns are merged with a logical OR. |
| 10 | Phase 2 | Single-sample diseases were removed in two separate cells and the label encoder was fitted twice (773 then 754 classes). | One cleaning step, one encoder. |
| 11 | Phase 2 | Symptoms that never occur keep random initial DNN weights, so selecting one in the app would add noise. | Symptoms never present in the training split are removed and listed in the metadata. |
| 12 | Dataset loader | All symptom columns loaded as `int64` (726 MB in memory). | Loaded as `uint8`. |
| 13 | Phase 2, Phase 6 | No validation set. `validation_split=0.10` used the last 10% of the training array, and there was nothing to tune the ensemble on without touching the test set. | Explicit train, validation and test splits. Every class is guaranteed at least one training record. |
| 14 | Phase 2 | Class weights were computed and never used. | Removed. |
| 15 | Phase 1 | Duplicate and conflicting records were not examined. | New analysis of exact duplicates and identical symptom patterns with different diagnoses (an accuracy ceiling). |
| 16 | Architecture notes | Record and class counts in the text did not match the data. | Counts are printed from the data at run time. |

## Intelligence layers

| # | Where | Problem | Fix |
|---|---|---|---|
| 17 | Phase 3 | Emergency scores used a slow row-by-row `DataFrame.apply`. Stroke signs, gastrointestinal bleeding and airway swelling were not screened. | Vectorised scoring. Added red flags that exist in the dataset (`difficulty breathing`, `vomiting blood`, `slurring words`, `focal weakness`, `hemoptysis`, `melena`, `swollen tongue`). All weights are editable in `config.py`. |
| 18 | Phase 4 | The rule engine profiled only the 20 most frequent diseases, so 734 diseases could never match (the panic-disorder test patient returned "esophagitis 1"). Profiles were built from the full dataset, leaking test records. | Profiles for every disease from the training split only, an F1-based key-symptom score, and top-k evaluation on the test set. |
| 19 | Phase 8 | `DNN_WEIGHT = 0.80` and `ET_WEIGHT = 0.20` were defined but never used; no ensemble existed. | Real weighted ensemble of DNN and Naive Bayes, with the weight chosen on validation data and evaluated once on test data. |
| 20 | Phase 9 | The "explanation" only listed the reported symptoms. | Occlusion attribution: each symptom is removed in turn and its signed share of the evidence is measured (log-probability based, so it still works near 100% probability). |
| 21 | Phase 11 | Bug: `clinical_assistant(predicted_disease, confidence, level)` passed the confidence level as the emergency level, printing "Emergency Level : VERY HIGH". | Emergency and confidence are separate typed objects. A unit test guards against the swap. |
| 22 | Phase 11 | "Prediction highly reliable" is unsafe wording for a screening tool. | Safety-first recommendations: triage action, emergency numbers, few-symptom, rare-disease and model-disagreement warnings, and a medicine caution. |
| 23 | Phase 10 | Confidence thresholds were never checked. | Accuracy per confidence level on the test set, shown in the app. |
| 24 | Phase 12 | The report generator read global variables and could only describe one fixed patient. | `build_pdf_report(result, patient)` produces the PDF for any patient; the same function runs in the app. |
| 25 | Phase 13 | The classification report showed only numeric class ids. | Per-disease metrics with names, hardest diseases and most-confused pairs, saved as CSV. |
| 26 | Roadmap | Phase 14 (universal prediction pipeline) was never implemented and the notebook jumped from Phase 13 to 15. | `HealthBandhuPredictor` is the universal pipeline. Phase 14 reloads all files from disk and verifies identical predictions. |

## Structure

| # | Problem | Fix |
|---|---|---|
| 27 | Everything lived in one notebook, so the app would have needed a second copy of every function. | Shared `healthbandhu` package used by both the notebook and `App.py`. |
| 28 | Repeated imports, redundant cells (for example `df_raw.shape` with no output, repeated `uint8` to `float32` conversions) and an empty last cell. | Removed; every cell has one clear purpose and a heading. |
