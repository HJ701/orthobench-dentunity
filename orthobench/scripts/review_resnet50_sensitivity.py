#!/usr/bin/env python3
"""Sensitivity of the headline 33-entry statistics to the fine-tuned ResNet-50 entry (post-plan).

Plan amendment 2026-11-18 'C3b_leaderboard_sensitivity'. Stored per-patient predictions only (CPU).
Computes every statistic twice: with the stored 33 entries ('published', must reproduce
results/zoo_reference.json, label_version_null.json and rank_stability.json) and with the
'ResNet-50 / fine-tuned' entry replaced by the three CPU seeds of amendment C3 ('cpu_resnet50').
The published leaderboard is not changed.

  python scripts/review_resnet50_sensitivity.py --workbook W  -> results/review_resnet50_sensitivity.json
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import run_benchmark as R  # noqa: E402
import label_null as LN  # noqa: E402
from orthobench import zoo, variables as V  # noqa: E402

KEY = "ResNet-50 / fine-tuned"
LEADER = "DINOv2-S / fine-tuned"


def null_stats(ent, src, adj, n):
    """scripts/label_null.py, unchanged procedure and seed."""
    obs = float(kendalltau(LN.board(ent, src)[:, 0], LN.board(ent, adj)[:, 0]).statistic)
    rng = np.random.default_rng(LN.SEED)
    out = {"observed_tau_b": obs}
    for nm, truth in (("source", src), ("adjudicated", adj)):
        wa, wb = V.draw_weights(n, LN.N_DRAWS, rng.integers(1 << 31)), V.draw_weights(n, LN.N_DRAWS, rng.integers(1 << 31))
        ba, bb = LN.board(ent, truth, wa), LN.board(ent, truth, wb)
        out[f"N1_bootstrap_pairs_{nm}"] = LN.summ([kendalltau(ba[:, i], bb[:, i]).statistic for i in range(LN.N_DRAWS)], obs)
    base = LN.board(ent, src)[:, 0]
    taus = []
    for _ in range(LN.N_DRAWS):
        t = src.copy()
        idx = rng.choice(n, LN.N_CHANGE, replace=False)
        t[idx] = (t[idx] + rng.integers(1, 3, LN.N_CHANGE)) % 3
        taus.append(kendalltau(base, LN.board(ent, t)[:, 0]).statistic)
    out["N2_random_relabel_64"] = LN.summ(taus, obs)
    return out


def stats(ent, src, adj, w):
    names, preds = list(ent), list(ent.values())
    rec = {}
    for metric, nm in ((0, "macro_f1"), (1, "balanced_accuracy")):
        s = V.leaderboard_shift(names, preds, src, adj, w, metric)
        rec[nm] = {k: s[k] for k in ("kendall_tau_point", "kendall_tau_ci", "top_ref_a", "top_ref_b",
                                     "p_same_top_over_draws", "n_pairs", "n_flips", "n_significant_shifts",
                                     "n_significant_flips")}
        rec[nm]["n_significant_flips_with_1nn"] = sum(
            1 for p in s["pairs"] if p["flip"] and p["significant_shift"] and ("knn1" in p["a"] or "knn1" in p["b"]))
        rec[nm]["fine_tuned_ranks"] = {m["name"]: {"source": m["rank_ref_a"], "adjudicated": m["rank_ref_b"],
                                                   "score_source": m["score_ref_a"], "score_adjudicated": m["score_ref_b"]}
                                       for m in s["models"] if m["name"].endswith("fine-tuned")}
    ba = np.stack([V.entry_scores(src, p, w, 0) for p in preds], 1)
    bb = np.stack([V.entry_scores(adj, p, w, 0) for p in preds], 1)
    k = names.index(LEADER)
    ta, tb = ba.argmax(1) == k, bb.argmax(1) == k
    rec["leader_p_rank1"] = {"source": float(ta.mean()), "adjudicated": float(tb.mean()), "both": float((ta & tb).mean())}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workbook", type=Path, required=True)
    a = ap.parse_args()
    ctx = zoo.context(R.read_cases(a.workbook), R.CACHE)
    store = R._load_preds()
    test = np.array(sorted(str(x) for x in store["intraoral/dinov2/logreg#ids"]))
    src, adj, _ = zoo.reference_labels(ctx, test)
    w = V.draw_weights(len(test), R.N_BOOT, R.SEED)
    pub = LN.entries(store, test)
    assert len(pub) == 33 and KEY in pub

    def aligned(path):
        z = np.load(path, allow_pickle=True)
        pos = {str(p): i for i, p in enumerate(z["ids"])}
        return np.stack([z["prob"][pos[p]] for p in test]).argmax(1)

    cpu = dict(pub)
    cpu[KEY] = [aligned(R.ROOT / f"data/preds/review_resnet50_protocol_cpu_s{s}.npz") for s in range(3)]
    out = {"provenance": "run", "status": "post-plan sensitivity",
           "plan_amendment": "experiments/analysis_plan.json amendments[C3b_leaderboard_sensitivity]",
           "replaced_entry": KEY, "n_test": len(test), "bootstrap_replicates": R.N_BOOT, "bootstrap_seed": R.SEED}
    for nm, ent in (("published", pub), ("cpu_resnet50", cpu)):
        rec = stats(ent, src, adj, w)
        rec["label_null"] = null_stats(ent, src, adj, len(test))
        out[nm] = rec
        f = rec["macro_f1"]
        print(f"[{nm}] tau {f['kendall_tau_point']:.4f} {f['kendall_tau_ci']} flips {f['n_flips']} sig {f['n_significant_shifts']} "
              f"sigflips {f['n_significant_flips']} (1-NN {f['n_significant_flips_with_1nn']}) top {f['top_ref_a']} / {f['top_ref_b']} "
              f"leader p1 {rec['leader_p_rank1']} N2 median {rec['label_null']['N2_random_relabel_64']['median']:.4f}", flush=True)
    R.save("review_resnet50_sensitivity.json", out)


if __name__ == "__main__":
    main()
