"""Port McNemar pairwise tests + composite ranking to the per-dataset ft40 study.

Reads the saved artifacts of the ft40 notebook study (split indices, per-model
test-set predictions, metrics.json) and writes, for each dataset and each
holdout scheme:

    artifacts/ft40/<ds>/results/significance/mcnemar_<scheme>.csv
    artifacts/ft40/<ds>/results/composite_<scheme>.csv

Uses the exact same implementations as the combined study (ids.evaluation.
significance.mcnemar_pairwise, ids.selection.pick_best.composite_score) so the
numbers are methodologically identical, just computed per dataset.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ids import config
from ids.evaluation.significance import mcnemar_pairwise
from ids.models.registry import MODEL_NAMES
from ids.selection.pick_best import composite_score

SCHEMES = ["70_30", "80_20", "90_10", "70_20_10"]


def main() -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout)
    for ds in config.VALID_DATASETS:
        root = config.ARTIFACTS_DIR / "ft40" / ds
        y = pd.read_parquet(root / "data" / "clean.parquet")["y"].to_numpy()
        sig_dir = root / "results" / "significance"
        sig_dir.mkdir(parents=True, exist_ok=True)

        for scheme in SCHEMES:
            test_idx = np.load(root / "splits" / f"{scheme}_indices.npz")["test_idx"]
            y_true = y[test_idx]

            preds = {m: np.load(root / "runs" / scheme / m / "predictions.npy")
                     for m in MODEL_NAMES}
            mc = mcnemar_pairwise(preds, y_true)
            mc.insert(0, "dataset", ds)
            mc.insert(1, "scheme", scheme)
            mc.to_csv(sig_dir / f"mcnemar_{scheme}.csv", index=False)
            n_sig = int(mc["significant_0.05"].sum())
            logging.info("%s %s: %d/%d pairs significant at 0.05", ds, scheme,
                         n_sig, len(mc))

            metrics = {m: json.loads((root / "runs" / scheme / m /
                                      "metrics.json").read_text())["test"]
                       for m in MODEL_NAMES}
            max_time = max(v["detection_time_ms"] for v in metrics.values()) or 1.0
            rows = [{
                "dataset": ds, "scheme": scheme, "model": m,
                "f1_macro": v["f1_macro"], "roc_auc": v.get("roc_auc"),
                "detection_time_ms": v["detection_time_ms"],
                "composite": round(composite_score(
                    v["f1_macro"], v.get("roc_auc"),
                    v["detection_time_ms"], max_time), 4),
            } for m, v in metrics.items()]
            comp = (pd.DataFrame(rows)
                    .sort_values("composite", ascending=False, ignore_index=True))
            comp.insert(3, "rank", range(1, len(comp) + 1))
            comp.to_csv(root / "results" / f"composite_{scheme}.csv", index=False)
            logging.info("%s %s: best by composite = %s (%.4f)", ds, scheme,
                         comp.iloc[0]["model"], comp.iloc[0]["composite"])


if __name__ == "__main__":
    main()
