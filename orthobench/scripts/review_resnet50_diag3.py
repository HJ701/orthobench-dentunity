#!/usr/bin/env python3
"""Post-plan per-tensor gradient check, MPS versus CPU, one training batch (ResNet-50; G1: DINOv2-S, ConvNeXt-T).

Plan amendment 2026-11-17 'C2d_per_tensor_gradients' in ../analysis_plan.json, logged before running.
Training images only; nothing scored on test; nothing selected.

  python scripts/review_resnet50_diag3.py --workbook W   -> results/review_runs/resnet50_diag3.json
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from orthobench import finetune, zoo  # noqa: E402
from orthobench.reference_audit import read_cases  # noqa: E402
from finetune_reference import PixelReader  # noqa: E402


def main():
    import torch
    from orthobench.features import _IN_MEAN, _IN_STD
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workbook", required=True, type=Path)
    ap.add_argument("--arch", default="resnet50", choices=["resnet50", "dinov2", "convnext_tiny"])
    ap.add_argument("--modes", nargs="+", default=["train", "eval"])
    a = ap.parse_args()
    out = ROOT / ("results/review_runs/resnet50_diag3.json" if a.arch == "resnet50"
                  else f"results/review_runs/gradcheck_{a.arch}" + ("" if a.modes != ["eval"] else "_eval") + ".json")
    cache = ROOT / "data/feat_cache/pixels_intraoral_256.u8"
    meta = dict(np.load(str(cache) + ".meta.npz", allow_pickle=True))
    arr = PixelReader(str(cache), int(meta["n"]))
    ctx = zoo.context(read_cases(a.workbook), ROOT / "data/feat_cache")
    pids = meta["patient_id"].astype(str)
    y = meta["label_numeric"].astype(int)
    tr = np.where(np.array([p not in ctx["audit_ids"] and p not in ctx["quarantine"] for p in pids]))[0]
    cnt = np.bincount(y[tr], minlength=3).astype(np.float64)
    w = torch.tensor(cnt.sum() / (3 * np.maximum(cnt, 1)), dtype=torch.float32)
    sel = np.sort(tr[np.random.RandomState(0).permutation(len(tr))][:16])
    mean = torch.tensor(_IN_MEAN).view(1, 3, 1, 1); std = torch.tensor(_IN_STD).view(1, 3, 1, 1)
    x = finetune._prep(arr, sel, True, np.random.RandomState(0), mean, std, "cpu")
    t = torch.from_numpy(y[sel]).long()
    torch.manual_seed(0)
    net0, head0 = finetune._model("cpu", arch=a.arch)
    res = {"provenance": "run", "status": "post-plan diagnostic (one training batch)",
           "plan_amendment": "experiments/analysis_plan.json amendments[" + ("C2d_per_tensor_gradients" if a.arch == "resnet50"
                                                                         else "G1_gradcheck_other_backbones") + "]",
           "arch": a.arch, "modes": {}}
    for mode in a.modes:
        g, feats, losses = {}, {}, {}
        for dev in ("cpu", "mps"):
            net, head = copy.deepcopy(net0).to(dev), copy.deepcopy(head0).to(dev)
            (net.train() if mode == "train" else net.eval()); head.train()
            f = net(x.to(dev)); z = head(f)
            loss = torch.nn.CrossEntropyLoss(weight=w.to(dev))(z, t.to(dev))
            loss.backward()
            losses[dev] = float(loss.detach())
            feats[dev] = f.detach().cpu().numpy().astype(np.float64)
            g[dev] = {nm: p.grad.detach().cpu().numpy().astype(np.float64).ravel().copy() for nm, p in
                      list(net.named_parameters()) + [("head." + k, v) for k, v in head.named_parameters()]
                      if p.grad is not None}
            del net, head
        rows = []
        for nm in g["cpu"]:
            c, m = g["cpu"][nm], g["mps"][nm]
            nc, nmps = float(np.linalg.norm(c)), float(np.linalg.norm(m))
            rows.append({"tensor": nm, "norm_cpu": nc, "norm_mps": nmps,
                         "rel_diff": float(np.linalg.norm(m - c) / nc) if nc > 0 else None,
                         "cosine": float(c @ m / (nc * nmps)) if nc > 0 and nmps > 0 else None})
        rels = [r["rel_diff"] for r in rows if r["rel_diff"] is not None]
        res["modes"][mode] = {"loss_cpu": losses["cpu"], "loss_mps": losses["mps"],
                              "feature_rel_diff": float(np.linalg.norm(feats["mps"] - feats["cpu"]) / np.linalg.norm(feats["cpu"])),
                              "median_rel_diff": float(np.median(rels)), "max_rel_diff": float(np.max(rels)),
                              "n_tensors_rel_diff_above_1e-2": int(sum(r > 1e-2 for r in rels)), "n_tensors": len(rels),
                              "tensors": rows}
        top = sorted([r for r in rows if r["rel_diff"] is not None], key=lambda r: -r["rel_diff"])[:6]
        print(f"[diag3] {mode}: loss cpu {losses['cpu']:.6f} mps {losses['mps']:.6f} feat-rel "
              f"{res['modes'][mode]['feature_rel_diff']:.2e} median-rel {res['modes'][mode]['median_rel_diff']:.2e} "
              f">1e-2: {res['modes'][mode]['n_tensors_rel_diff_above_1e-2']}/{len(rels)}", flush=True)
        for r in top + [r for r in rows if r["tensor"].startswith("head.")]:
            print(f"   {r['tensor']:28s} |g| cpu {r['norm_cpu']:.3e} mps {r['norm_mps']:.3e} rel {r['rel_diff']:.2e} cos {r['cosine']:.4f}", flush=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
