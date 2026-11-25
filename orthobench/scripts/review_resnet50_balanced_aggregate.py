#!/usr/bin/env python3
"""Aggregate the post-plan class-balanced ResNet-50 arms next to the protocol and warmup entries.

Plan amendment 2026-11-17 'C3_resnet50_protocol_cpu'. Entry = mean of the seeds' scores on the shared
2,000 patient-bootstrap draws (seed 20261108), as for every fine-tuned entry; paired differences
use the same draws. Per seed: macro-F1, per-class recall, confusion matrix (rows = label,
columns = prediction, order I/II/III), predicted-class counts, and two descriptive checks of
whether the network separates the classes at all: the largest predicted probability of any
class over the 189 test patients and the one-vs-rest AUROC of the Class III probability.
Nothing is refitted and nothing is selected.

  python scripts/review_resnet50_balanced_aggregate.py --workbook W  -> results/review_resnet50_balanced.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import run_benchmark as R  # noqa: E402
from orthobench import zoo, variables as V  # noqa: E402
from review_resnet50_balanced import ARMS  # noqa: E402

SEEDS = (0, 1, 2)


def per_class(truth, pred):
    cm = np.zeros((3, 3), int)
    np.add.at(cm, (truth, pred), 1)
    rec = [float(cm[c, c] / cm[c].sum()) if cm[c].sum() else None for c in range(3)]
    return cm.tolist(), rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workbook", type=Path, required=True)
    a = ap.parse_args()
    ctx = zoo.context(R.read_cases(a.workbook), R.CACHE)
    store = R._load_preds()
    test = np.array(sorted(str(x) for x in store["intraoral/dinov2/logreg#ids"]))
    manifest, final, _ = zoo.reference_labels(ctx, test)
    labels = {"source": manifest, "adjudicated": final}
    w = V.draw_weights(len(test), R.N_BOOT, R.SEED)

    def aligned(path):
        z = np.load(path, allow_pickle=True)
        pos = {str(p): i for i, p in enumerate(z["ids"])}
        return np.stack([z["prob"][pos[p]] for p in test])

    probs = {"protocol": {s: R._align(store, f"finetune/resnet50/s{s}", test) for s in SEEDS},
             "warmup": {s: aligned(R.ROOT / f"data/preds/review_resnet50_warmup_s{s}.npz") for s in SEEDS}}
    meta = {"protocol": {}, "warmup": {s: json.loads((R.RESULTS / f"review_runs/resnet50_warmup_s{s}.json").read_text())
                                       for s in SEEDS}}
    for arm in ARMS:
        done = [s for s in SEEDS if (R.RESULTS / f"review_runs/resnet50_{arm}_s{s}.json").exists()]
        probs[arm] = {s: aligned(R.ROOT / f"data/preds/review_resnet50_{arm}_s{s}.npz") for s in done}
        meta[arm] = {s: json.loads((R.RESULTS / f"review_runs/resnet50_{arm}_s{s}.json").read_text()) for s in done}
    lead = [R._align(store, f"finetune/dinov2/s{s}", test).argmax(1) for s in SEEDS]
    zr = json.loads((R.RESULTS / "zoo_reference.json").read_text())["macro_f1"]["models"]

    out = {"provenance": "run", "status": "post-plan",
           "plan_amendment": "experiments/analysis_plan.json amendments[C3_resnet50_protocol_cpu]",
           "n_test": len(test), "test_label_counts_I_II_III": {r: np.bincount(t, minlength=3).tolist() for r, t in labels.items()},
           "bootstrap_replicates": R.N_BOOT, "bootstrap_seed": R.SEED,
           "arms": {k: ARMS[k]["description"] for k in ARMS}, "groups": {}}
    boots = {}
    for g, pr in probs.items():
        seeds = sorted(pr)
        if not seeds:
            continue
        preds = [pr[s].argmax(1) for s in seeds]
        rec = {"seeds_done": seeds, "complete": len(seeds) == len(SEEDS), "per_seed": []}
        for s, p in zip(seeds, preds):
            row = {"seed": s, "pred_class_counts_I_II_III": np.bincount(p, minlength=3).tolist(),
                   "max_prob_any_class": float(pr[s].max()), "max_prob_class_III": float(pr[s][:, 2].max())}
            for r, t in labels.items():
                cm, rc = per_class(t, p)
                f1, bacc = V.scores(t, p)
                row[r] = {"macro_f1": float(f1[0]), "balanced_accuracy": float(bacc[0]), "recall_I_II_III": rc,
                          "confusion_rows_label_cols_pred": cm, "auroc_class_III": V.auroc(pr[s][:, 2], t == 2)}
            m = meta[g].get(s, {})
            for k in ("train_loss_per_epoch", "wall_seconds"):
                if k in m:
                    row[k] = m[k]
            rec["per_seed"].append(row)
        for r, t in labels.items():
            b = V.entry_scores(t, preds, w, 0)
            boots[(g, r)] = b
            s0 = float(V.entry_scores(t, preds)[0])
            others = sorted((m["score_ref_a" if r == "source" else "score_ref_b"] for m in zr), reverse=True)
            rec[r] = {"macro_f1": s0, "macro_f1_ci": V.ci(b),
                      "balanced_accuracy": float(V.entry_scores(t, preds, None, 1)[0]),
                      "balanced_accuracy_ci": V.ci(V.entry_scores(t, preds, w, 1)),
                      "mean_recall_I_II_III": np.mean([per_class(t, p)[1] for p in preds], 0).tolist(),
                      "leader_minus_entry": float(V.entry_scores(t, lead)[0] - s0),
                      "leader_minus_entry_ci": V.ci(V.entry_scores(t, lead, w, 0) - b),
                      "rank_if_inserted_among_33": int(1 + sum(x > s0 for x in others))}
        out["groups"][g] = rec
    out["paired_differences"] = {}
    for arm in ARMS:
        if not out["groups"].get(arm, {}).get("complete"):
            continue
        for ref in ("protocol", "warmup"):
            for r in labels:
                d = boots[(arm, r)] - boots[(ref, r)]
                out["paired_differences"][f"{arm}_minus_{ref}_{r}"] = {
                    "diff": out["groups"][arm][r]["macro_f1"] - out["groups"][ref][r]["macro_f1"],
                    "ci": V.ci(d), "share_of_draws_above_zero": float((d > 0).mean())}
    R.save("review_resnet50_balanced.json", out)


if __name__ == "__main__":
    main()
