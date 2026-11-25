#!/usr/bin/env python3
"""Post-plan reruns of fine-tuned ResNet-50 on the reference split (device control of the protocol entry).

Plan amendment 2026-11-17 'C3_resnet50_protocol_cpu' in ../analysis_plan.json, logged before running.
Same split, pixel cache, augmentation, batch 16, 8 epochs, head rate 1e-3, weight decay 0.05 and
patient aggregation as the protocol (scripts/finetune_reference.py) and C1. The protocol and C1 losses are
already inverse-frequency-weighted cross-entropy with weights from the training images; the arms
below change only what ARMS states. One seed per call; resumable (skips a seed whose result file
exists). Predictions go to the gitignored data/preds/, never into the 33-entry store.

  python scripts/review_resnet50_balanced.py --workbook W --arm protocol_cpu --seed 0
-> results/review_runs/resnet50_<arm>_s<seed>.json, data/preds/review_resnet50_<arm>_s<seed>.npz
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

BATCH, EPOCHS = 16, 8
# name -> recipe; every arm keeps head lr 1e-3, weight decay 0.05, batch 16, 8 epochs.
ARMS = {
    "protocol_cpu": {"device": "cpu", "lr_backbone": 1e-4, "warmup": False, "balanced_sampler": False,
                     "class_weighted_loss": True,
                     "description": "protocol recipe unchanged (inverse-frequency-weighted cross-entropy from the "
                                    "training images, backbone lr 1e-4, head lr 1e-3, cosine from step 1, no warmup, "
                                    "batch 16, 8 epochs), run on CPU instead of MPS"},
}


def main():
    from orthobench import finetune, zoo, variables as V
    from orthobench.reference_audit import read_cases
    from finetune_reference import PixelReader
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workbook", required=True, type=Path)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    arm = ARMS[a.arm]
    out = ROOT / f"results/review_runs/resnet50_{a.arm}_s{a.seed}.json"
    if out.exists():
        print(f"[skip] {out.name} exists"); return
    cache = ROOT / "data/feat_cache/pixels_intraoral_256.u8"
    meta = dict(np.load(str(cache) + ".meta.npz", allow_pickle=True))
    arr = PixelReader(str(cache), int(meta["n"]))
    ctx = zoo.context(read_cases(a.workbook), ROOT / "data/feat_cache")
    pids = meta["patient_id"].astype(str)
    y = meta["label_numeric"].astype(int)
    tr_mask = np.array([p not in ctx["audit_ids"] and p not in ctx["quarantine"] for p in pids])
    te_mask = np.array([p in ctx["test_ids"] and p not in ctx["quarantine"] for p in pids])
    assert not (set(pids[tr_mask]) & (ctx["audit_ids"] | ctx["quarantine"] | set(pids[te_mask])))
    tr, te = np.where(tr_mask)[0], np.where(te_mask)[0]
    test = np.array(sorted(set(pids[te])))
    manifest, final, _ = zoo.reference_labels(ctx, test)
    warmup = len(tr) // BATCH + 1 if arm["warmup"] else 0
    losses = []

    def log(msg):
        print(msg, flush=True)
        m = re.search(r"epoch (\d+)/\d+ loss ([0-9.]+)", msg)
        if m:
            losses.append(float(m.group(2)))

    t0 = time.monotonic()
    prob = finetune.train_eval(arr, meta, tr, te, seed=a.seed, epochs=EPOCHS, bs=BATCH,
                               lr_bb=arm["lr_backbone"], arch="resnet50", warmup_steps=warmup,
                               device=arm["device"], log=log, balanced_sampler=arm["balanced_sampler"],
                               class_weighted_loss=arm["class_weighted_loss"])
    agg = np.stack([prob[pids[te] == p].mean(0) for p in test])
    pred_dir = ROOT / "data/preds"; pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez(pred_dir / f"review_resnet50_{a.arm}_s{a.seed}.npz", prob=agg, ids=test)
    pred = agg.argmax(1)
    cnt = np.bincount(y[tr], minlength=3)
    res = {"provenance": "run", "status": "post-plan",
           "plan_amendment": "experiments/analysis_plan.json amendments[C3_resnet50_protocol_cpu]",
           "arch": "resnet50", "arm": a.arm, "recipe": {k: v for k, v in arm.items()}, "seed": a.seed,
           "lr_head": 1e-3, "batch": BATCH, "epochs": EPOCHS, "warmup_steps": warmup,
           "train_image_counts_I_II_III": cnt.tolist(),
           "inverse_frequency_weights": (cnt.sum() / (3 * cnt)).tolist(),
           "n_train_images": int(len(tr)), "n_train_patients": int(len(set(pids[tr]))),
           "n_test_patients": int(len(test)), "train_loss_per_epoch": losses,
           "pred_class_counts_I_II_III": np.bincount(pred, minlength=3).tolist(),
           "manifest_macro_f1": float(V.scores(manifest, pred)[0][0]),
           "final_macro_f1": float(V.scores(final, pred)[0][0]),
           "final_balanced_accuracy": float(V.scores(final, pred)[1][0]),
           "wall_seconds": time.monotonic() - t0}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"[resnet50-{a.arm}] s{a.seed}: source {res['manifest_macro_f1']:.3f} adjudicated "
          f"{res['final_macro_f1']:.3f} counts {res['pred_class_counts_I_II_III']} ({res['wall_seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
