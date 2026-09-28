#!/bin/zsh
# Execute the ft40 notebook study end-to-end: both datasets, notebooks 01-05.
# Each dataset has its own permanent notebook set (notebooks/cic2017/,
# notebooks/cic2018/). Notebooks are executed IN PLACE so they keep their cell
# outputs as the archived record; an HTML copy of each goes to
# artifacts/ft40/<dataset>/report_html/. Notebook 04 skips any (scheme, model)
# with existing metrics.json, so re-running this script resumes an interrupted
# study instead of redoing it.
set -euo pipefail
BACKEND="$(cd "$(dirname "$0")/.." && pwd)"
PM="$BACKEND/.venv/bin/papermill"
JUP="$BACKEND/.venv/bin/jupyter"
LOG="$BACKEND/artifacts/ft40/run_all.log"
mkdir -p "$BACKEND/artifacts/ft40"
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

for DS in cic2017 cic2018; do
  NBDIR="$BACKEND/notebooks/$DS"
  HTML="$BACKEND/artifacts/ft40/$DS/report_html"
  mkdir -p "$HTML"
  for nb in 01_load_and_eda 02_preprocess 03_feature_selection 04_experiments 05_results; do
    echo "[$(ts)] START $DS $nb" | tee -a "$LOG"
    "$PM" --cwd "$NBDIR" -k ids "$NBDIR/$nb.ipynb" "$NBDIR/$nb.ipynb" >> "$LOG" 2>&1
    "$JUP" nbconvert --to html "$NBDIR/$nb.ipynb" --output-dir "$HTML" >> "$LOG" 2>&1
    echo "[$(ts)] END   $DS $nb" | tee -a "$LOG"
  done
done
echo "[$(ts)] ALL DONE" | tee -a "$LOG"
