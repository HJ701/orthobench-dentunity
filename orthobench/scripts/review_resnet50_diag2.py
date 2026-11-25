#!/usr/bin/env python3
"""Post-plan follow-up of the ResNet-50 training diagnostic: MPS versus CPU in lockstep.

Plan amendment 2026-11-17 'C2b_mps_cpu_lockstep' / 'C2c_second_order' in ../analysis_plan.json,
logged before running. Training images only; nothing is scored on test; nothing is selected.

  python scripts/review_resnet50_diag2.py --workbook W   -> results/review_runs/resnet50_diag2.json
"""
import argparse
import copy
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
import review_resnet50_diag as D  # noqa: E402

LOCK_STEPS, BATCH = 30, 16


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def lockstep(arr, y, batches, w):
    import torch
    from orthobench.features import _IN_MEAN, _IN_STD
    torch.manual_seed(0)
    net_c, head_c = finetune._model("cpu", arch="resnet50")
    net_m, head_m = copy.deepcopy(net_c).to("mps"), copy.deepcopy(head_c).to("mps")
    mean = torch.tensor(_IN_MEAN).view(1, 3, 1, 1); std = torch.tensor(_IN_STD).view(1, 3, 1, 1)
    models = {"cpu": (net_c, head_c), "mps": (net_m, head_m)}
    opts, losses = {}, {}
    for dev, (n, h) in models.items():
        n.train(); h.train()
        opts[dev] = torch.optim.AdamW([{"params": n.parameters(), "lr": 1e-4},
                                       {"params": h.parameters(), "lr": 1e-3}], weight_decay=0.05)
        losses[dev] = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to(dev))
    rng = np.random.RandomState(0)
    rows = []
    for k, sel in enumerate(batches[:LOCK_STEPS]):
        x = finetune._prep(arr, np.sort(sel), True, rng, mean, std, "cpu")
        t = torch.from_numpy(y[np.sort(sel)]).long()
        grads, lv = {}, {}
        for dev, (n, h) in models.items():
            opts[dev].zero_grad()
            loss = losses[dev](h(n(x.to(dev))), t.to(dev))
            loss.backward()
            lv[dev] = float(loss.detach())
            grads[dev] = [(nm, p.grad.detach().cpu().numpy().ravel()) for nm, p in
                          list(n.named_parameters()) + [("head." + a, b) for a, b in h.named_parameters()]]
        gc = np.concatenate([g for _, g in grads["cpu"]]); gm = np.concatenate([g for _, g in grads["mps"]])
        per = [(nm, rel(m, c)) for (nm, c), (_, m) in zip(grads["cpu"], grads["mps"]) if np.linalg.norm(c) > 0]
        bn = [r for nm, r in per if ".bn" in nm or "downsample.1" in nm]
        row = {"step": k + 1, "loss_cpu": lv["cpu"], "loss_mps": lv["mps"], "grad_rel_diff_total": rel(gm, gc),
               "grad_rel_diff_max_tensor": max(r for _, r in per), "grad_rel_diff_max_tensor_name": max(per, key=lambda z: z[1])[0],
               "grad_rel_diff_median_tensor": float(np.median([r for _, r in per])),
               "grad_rel_diff_max_bn": max(bn) if bn else None}
        for dev in models:
            opts[dev].step()
        pc = np.concatenate([p.detach().cpu().numpy().ravel() for p in list(net_c.parameters()) + list(head_c.parameters())])
        pm = np.concatenate([p.detach().cpu().numpy().ravel() for p in list(net_m.parameters()) + list(head_m.parameters())])
        row["param_rel_diff"] = rel(pm, pc)
        if k == 0:
            rc = np.concatenate([b.detach().cpu().numpy().ravel() for nm, b in net_c.named_buffers() if "running" in nm])
            rm = np.concatenate([b.detach().cpu().numpy().ravel() for nm, b in net_m.named_buffers() if "running" in nm])
            row["bn_running_stats_rel_diff"] = rel(rm, rc)
        rows.append(row)
        print(f"  lock {k+1:2d} loss cpu {lv['cpu']:.4f} mps {lv['mps']:.4f} grad-rel {row['grad_rel_diff_total']:.2e} "
              f"max-tensor {row['grad_rel_diff_max_tensor']:.2e} param-rel {row['param_rel_diff']:.2e}", flush=True)
    del net_c, head_c, net_m, head_m
    torch.mps.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workbook", required=True, type=Path)
    a = ap.parse_args()
    out = ROOT / "results/review_runs/resnet50_diag2.json"
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
    res = {"provenance": "run", "status": "post-plan diagnostic (training images only)",
           "plan_amendment": "experiments/analysis_plan.json amendments[C2b_mps_cpu_lockstep, C2c_second_order]"}
    order0 = tr[np.random.RandomState(0).permutation(len(tr))]
    t0 = time.monotonic()
    print("[diag2] lockstep", flush=True)
    res["lockstep"] = lockstep(arr, y, [order0[i * BATCH:(i + 1) * BATCH] for i in range(LOCK_STEPS)], w)
    res["lockstep_wall_seconds"] = time.monotonic() - t0
    order1 = tr[np.random.RandomState(1).permutation(len(tr))]
    batches1 = [order1[i * BATCH:(i + 1) * BATCH] for i in range(D.STEPS)]
    res["second_order"] = {}
    for name in ("a_mps_protocol_rates", "b_cpu_protocol_rates"):
        dev, lr_bb, frozen = D.ARMS[name]
        print(f"[diag2] order1 {name}", flush=True)
        res["second_order"][name] = D.run_arm(arr, y, batches1, dev, lr_bb, frozen, w)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
