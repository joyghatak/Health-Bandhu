# Dataset

The training notebook downloads the symptom dataset (about 190 MB, 246,945 records) from Google Drive to `data/medical_dataset.csv` on its first run.

To add it manually, place the CSV here with that exact name. The first column must be `diseases`; every other column is a 0/1 symptom indicator.

The dataset is not included in the release zip and is ignored by Git.
