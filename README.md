# Comparative Analysis of Machine Learning Algorithms for Network Intrusion Detection

Final-year project for the B.Sc. in Cyber Security at Redeemer's University (2026).
Five classifiers (Decision Tree, Random Forest, SVM, Naive Bayes and KNN) are compared on
the CIC-IDS-2017 and CSE-CIC-IDS2018 flow datasets, using three feature-selection methods,
several train/test split schemes, 5-fold cross-validation and McNemar significance tests.

This repository holds the Python package the notebooks call, the six notebook studies with
their executed outputs, and the metrics, tables and plots from the completed runs. Two things
are left out to keep it small: the raw datasets (about 7 GB) and the bulky run outputs
(fitted models, prediction and probability arrays, and split indices, about 3.7 GB).

## Layout

```
datasets/                    raw CSV files (not included; see "Getting the data")
backend/ids/                 Python package (loading, preprocessing, models, evaluation)
backend/notebooks/
  cic2017/, cic2018/         main study: top-40 features by Random Forest importance
  cic2017_mi/, cic2018_mi/   supplementary study: Mutual Information features
  cic2017_rfe/, cic2018_rfe/ supplementary study: SVM-RFE features
  html/                      each executed notebook exported to HTML, per study
  html_onepage/              all five notebooks of each main study in one HTML page
backend/artifacts/           outputs of the completed runs: metrics, tables, plots and logs
backend/scripts/             run_ft40.sh (runs a whole study unattended) and
                             ft40_significance.py (McNemar tests + composite ranking)
```

## Results without running anything

The notebooks in `backend/notebooks/` are saved with their outputs, and the same content
opens in any browser from `backend/notebooks/html/` and `html_onepage/`. The metrics,
tables and plots behind them are under `backend/artifacts/`.

## Getting the data

Download the CSV releases of both datasets from the Canadian Institute for Cybersecurity:

- CIC-IDS-2017: https://www.unb.ca/cic/datasets/ids-2017.html
- CSE-CIC-IDS2018: https://www.unb.ca/cic/datasets/ids-2018.html

Put the CSV files in these folders. The notebooks read every `.csv` file in each one, and
the code expects the space in the first folder name.

```
datasets/
  CIC-IDS- 2017/     Monday-WorkingHours.pcap_ISCX.csv, Tuesday-WorkingHours.pcap_ISCX.csv, ...
  CSE-CIC-IDS2018/   02-14-2018.csv, 02-15-2018.csv, ...
```

To keep the data somewhere else, set the `DATASETS_DIR` environment variable to that folder.

## Setup to run the experiments

Requires Python 3.11 and roughly 16 GB of RAM; a full study takes several hours,
most of it in notebook 04.

```
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]" jupyter papermill
python -m ipykernel install --user --name ids --display-name "IDS (.venv)"
```

## Running

Each study folder contains five notebooks. Open them in Jupyter and run top to
bottom in order: 01 (import and exploratory analysis), 02 (preprocessing),
03 (feature selection), 04 (training and evaluation across all split schemes),
05 (results collation). Every notebook logs each step and writes its outputs to
`backend/artifacts/`.

Three behaviours worth knowing:

- Runs resume. Notebook 04 skips any scheme and model whose `metrics.json` already
  exists, so an interrupted run continues where it stopped. This repository ships
  those files, so delete a study's folder under `backend/artifacts/` before
  re-running it (for example `backend/artifacts/ft40/cic2017`). Notebook 01 then
  rebuilds everything from the raw CSVs.
- The large data tables that notebooks 01 and 02 write (the imported and the
  cleaned dataset) are not kept. To re-run notebook 03, 04 or 05 on its own, run
  01 and 02 first so those tables are rebuilt from the CSVs.
- The supplementary studies reuse the main study's imported data automatically
  when it exists, so they skip the long CSV import.

To run both main studies unattended instead of interactively:

```
zsh backend/scripts/run_ft40.sh
```

After a study's notebook 04 has finished, the significance tests and composite
ranking are computed with:

```
backend/.venv/bin/python backend/scripts/ft40_significance.py
```

The script reads the prediction arrays and split indices that notebook 04 writes,
so it only works after a study has been run locally.
