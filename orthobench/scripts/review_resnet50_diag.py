#!/usr/bin/env python3
"""Post-plan training diagnostic for fine-tuned ResNet-50 (does it train at all in this pipeline?).

Plan amendment 2026-11-17 'C2_resnet50_train_diagnostic' in ../analysis_plan.json, logged before
running. Training patients only: no test image is read and nothing is scored. Same pixel cache,
augmentation, model construction, AdamW (weight decay 0.05) and class-weighted cross-entropy as
the protocol fine-tuning; one fixed sequence of 100 batches of 16 training images; constant rates.

  python scripts/review_resnet50_diag.py --workbook W   -> results/review_runs/resnet50_diag.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from orthobench import finetune, zoo  # noqa: E402
from orthobench.reference_audit import read_cases  # noqa: E402
from finetune_reference import PixelReader  # noqa: E402

STEPS, BATCH, EVERY = 100, 16, 10
ARMS = {"a_mps_protocol_rates": ("mps", 1e-4, False), "b_cpu_protocol_rates": ("cpu", 1e-4, False),
        "c_mps_c1_rates": ("mps", 1e-5, False), "d_mps_frozen_backbone": ("mps", 0.0, True)}


def run_arm(arr, y, batches, device, lr_bb, frozen, w):
    import torch
    from orthobench.features import _IN_MEAN, _IN_STD
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    net, head = finetune._model(device, arch="resnet50")
    mean = torch.tensor(_IN_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(_IN_STD).view(1, 3, 1, 1).to(device)
    lossf = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to(device))
    groups = [{"params": head.parameters(), "lr": 1e-3}]
    if not frozen:
        groups.insert(0, {"params": net.parameters(), "lr": lr_bb})
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    log, losses, t0 = [], [], time.monotonic()
    for k, sel in enumerate(batches):
        (net.eval() if frozen else net.train()); head.train()
        x = finetune._prep(arr, np.sort(sel), True, rng, mean, std, device)
        t = torch.from_numpy(y[np.sort(sel)]).long().to(device)
        opt.zero_grad()
        if frozen:
            with torch.no_grad():
                f = net(x)
        else:
            f = net(x)
        z = head(f)
        loss = lossf(z, t)
        loss.backward(); opt.step()
        losses.append(float(loss.detach()))
        if (k + 1) % EVERY == 0 or k == 0:
            fd = f.detach().float().cpu().numpy(); zd = z.detach().float().cpu().numpy()
            log.append({"step": k + 1, "mean_loss_last": float(np.mean(losses[-EVERY:])),
                        "feature_batch_std": float(fd.std(0).mean()), "feature_mean": float(fd.mean()),
                        "feature_zero_share": float((fd == 0).mean()),
                        "logit_spread": float((zd.max(1) - zd.min(1)).mean()),
                        "pred_counts_I_II_III": np.bincount(zd.argmax(1), minlength=3).tolist()})
            print(f"  step {k+1:3d} loss {log[-1]['mean_loss_last']:.4f} featstd {log[-1]['feature_batch_std']:.4f} "
                  f"zero {log[-1]['feature_zero_share']:.3f} spread {log[-1]['logit_spread']:.3f}", flush=True)
    out = {"device": device, "lr_backbone": lr_bb, "lr_head": 1e-3, "backbone_frozen": frozen,
           "steps": len(batches), "log": log, "wall_seconds": time.monotonic() - t0}
    del net, head, opt
    if device == "mps":
        torch.mps.empty_cache()
    return out


def main():
    import torch
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workbook", required=True, type=Path)
    a = ap.parse_args()
    out = ROOT / "results/review_runs/resnet50_diag.json"
    if out.exists():
        print(f"[skip] {out.name} exists"); return
    cache = ROOT / "data/feat_cache/pixels_intraoral_256.u8"
    meta = dict(np.load(str(cache) + ".meta.npz", allow_pickle=True))
    arr = PixelReader(str(cache), int(meta["n"]))
    ctx = zoo.context(read_cases(a.workbook), ROOT / "data/feat_cache")
    pids = meta["patient_id"].astype(str)
    y = meta["label_numeric"].astype(int)
    tr = np.where(np.array([p not in ctx["audit_ids"] and p not in ctx["quarantine"] for p in pids]))[0]
    cnt = np.bincount(y[tr], minlength=3).astype(np.float64)
    w = cnt.sum() / (3 * np.maximum(cnt, 1))
    order = tr[np.random.RandomState(0).permutation(len(tr))]
    batches = [order[i * BATCH:(i + 1) * BATCH] for i in range(STEPS)]
    res = {"provenance": "run", "status": "post-plan diagnostic (training images only)",
           "plan_amendment": "experiments/analysis_plan.json amendments[C2_resnet50_train_diagnostic]",
           "n_train_images": int(len(tr)), "train_image_counts_I_II_III": cnt.astype(int).tolist(),
           "class_weights": w.tolist(), "chance_loss_ln3": float(np.log(3)), "arms": {}}
    # MPS vs CPU eval-mode features on the first batch (pretrained weights, no update)
    from orthobench.features import _IN_MEAN, _IN_STD
    feats = {}
    for dev in ("cpu", "mps"):
        torch.manual_seed(0)
        net, _ = finetune._model(dev, arch="resnet50"); net.eval()
        mean = torch.tensor(_IN_MEAN).view(1, 3, 1, 1).to(dev); std = torch.tensor(_IN_STD).view(1, 3, 1, 1).to(dev)
        with torch.no_grad():
            feats[dev] = net(finetune._prep(arr, np.sort(batches[0]), False, None, mean, std, dev)).cpu().numpy()
        del net
    res["eval_feature_mps_vs_cpu"] = {"max_abs_diff": float(np.abs(feats["cpu"] - feats["mps"]).max()),
                                      "max_abs_cpu": float(np.abs(feats["cpu"]).max()),
                                      "cpu_feature_batch_std": float(feats["cpu"].std(0).mean())}
    print("[diag] eval features", res["eval_feature_mps_vs_cpu"], flush=True)
    for name, (dev, lr_bb, frozen) in ARMS.items():
        print(f"[diag] {name}", flush=True)
        res["arms"][name] = run_arm(arr, y, batches, dev, lr_bb, frozen, w)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
