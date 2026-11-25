"""Fit annotation-availability controls and decompose model rank reversals."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

PSEUDOCOUNT = 16
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 701


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def rank_comparisons(path, first, second):
    rows = read_rows(path)
    for column in (first, second):
        assert sorted(int(row[column]) for row in rows) == list(range(1, len(rows) + 1))
    totals = defaultdict(Counter)
    for a, b in combinations(rows, 2):
        a_head = a["entry"].rsplit(" / ", 1)[1]
        b_head = b["entry"].rsplit(" / ", 1)[1]
        key = "same_head" if a_head == b_head else "different_heads"
        crossed = (int(a[first]) - int(b[first])) * (int(a[second]) - int(b[second])) < 0
        for group in (key, "all"):
            totals[group]["pairs"] += 1
            totals[group]["reversals"] += int(crossed)
    return {key: dict(counts) for key, counts in sorted(totals.items())}


def size_features(row):
    width, height = int(row["width"]), int(row["height"])
    assert width > 0 and height > 0
    coarse = (math.floor(2 * math.log2(width)), math.floor(2 * math.log2(height)))
    return coarse, (width, height)


class GeometryClassifier:
    """Shrink exact-size event rates towards coarser resolution groups."""
    def __init__(self, training):
        self.prior = sum(label for _, label in training) / len(training)
        self.coarse = defaultdict(Counter)
        self.exact = defaultdict(Counter)
        for row, label in training:
            coarse, exact = size_features(row)
            self.coarse[coarse][label] += 1
            self.exact[exact][label] += 1

    @staticmethod
    def smooth(counts, parent):
        return (counts[1] + PSEUDOCOUNT * parent) / (sum(counts.values()) + PSEUDOCOUNT)

    def predict(self, row):
        coarse, exact = size_features(row)
        coarse_rate = self.smooth(self.coarse.get(coarse, Counter()), self.prior)
        return self.smooth(self.exact.get(exact, Counter()), coarse_rate)


def ap_and_prevalence(labels, weights, levels):
    """Non-interpolated AP, pooling all tied scores at the same threshold."""
    totals = np.bincount(levels, weights=weights)
    events = np.bincount(levels, weights=labels * weights)
    if events.sum() == 0 or totals.sum() == 0:
        return None
    positives_above = events[::-1].cumsum()
    cases_above = totals[::-1].cumsum()
    precision = np.divide(positives_above, cases_above, out=np.zeros_like(cases_above),
                          where=cases_above > 0)
    ap = float(np.dot(precision, events[::-1]) / events.sum())
    return ap, float(events.sum() / totals.sum())


def summarize(model, sample, bootstrap):
    rows, outcomes = zip(*sample)
    labels = np.asarray(outcomes, dtype=float)
    scores = np.asarray([model.predict(row) for row in rows])
    _, levels = np.unique(scores, return_inverse=True)
    lineages, groups = np.unique([row["lineage"] for row in rows], return_inverse=True)
    observed = ap_and_prevalence(labels, np.ones(len(rows)), levels)
    assert observed is not None
    ap, prevalence = observed
    result = dict(images=len(rows), positives=int(labels.sum()), lineages=len(lineages),
                  ap=ap, constant_ap=prevalence, difference=ap - prevalence,
                  distinct_scores=len(np.unique(scores)),
                  familiar_exact_size=sum(size_features(row)[1] in model.exact for row in rows))
    if bootstrap:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        diffs = []
        for _ in range(BOOTSTRAP_DRAWS):
            draws = rng.integers(len(lineages), size=len(lineages))
            weights = np.bincount(draws, minlength=len(lineages))[groups]
            resampled = ap_and_prevalence(labels, weights, levels)
            if resampled is not None:
                diffs.append(resampled[0] - resampled[1])
        result["valid_resamples"] = len(diffs)
        result["difference_interval"] = np.quantile(diffs, [0.025, 0.975]).tolist()
    return result


def fit_task(sample):
    by_split = defaultdict(list)
    assignments = {}
    seen = set()
    for row, label in sample:
        assert row["sha256"] not in seen
        seen.add(row["sha256"])
        assert assignments.setdefault(row["lineage"], row["split"]) == row["split"]
        by_split[row["split"]].append((row, label))
    assert set(by_split) == {"train", "val", "test"}
    model = GeometryClassifier(by_split["train"])
    return dict(train_images=len(by_split["train"]), train_positives=sum(y for _,y in by_split["train"]),
                validation=summarize(model, by_split["val"], False),
                test=summarize(model, by_split["test"], True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("orthobench", type=Path)
    parser.add_argument("dentunity", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = args.dentunity / "dataset" / "manifests"
    detected = {r["sha256"]: r["split"] for r in read_rows(directory / "io_det.csv")}
    photos = read_rows(directory / "io_cls.csv")
    assert all(detected.get(r["sha256"], r["split"]) == r["split"] for r in photos)
    panos = [r for r in read_rows(directory / "px.csv") if r["role"] == "canonical"]
    assert {r["in_seg"] for r in panos} == {"0","1"}
    tables = args.orthobench / "dataset" / "leaderboard"
    result = {
        "design": {"pseudocount": PSEUDOCOUNT, "lineage_draws": BOOTSTRAP_DRAWS,
                   "random_seed": BOOTSTRAP_SEED, "numpy": np.__version__},
        "photo_detection_membership": fit_task([(r, int(r["sha256"] in detected)) for r in photos]),
        "panoramic_mask_absence": fit_task([(r, int(r["in_seg"] == "0")) for r in panos]),
        "rank_reversals": {
            "reference": rank_comparisons(tables / "reference_arm.csv", "rank_source", "rank_adjudicated"),
            "split": rank_comparisons(tables / "split_arm.csv", "rank_patient_split", "rank_image_split"),
        },
    }
    contents = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(contents, end="")
    else:
        args.output.write_text(contents, encoding="utf-8")


if __name__ == "__main__":
    main()
