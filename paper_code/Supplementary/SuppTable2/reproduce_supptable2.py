#!/usr/bin/env python3
"""
Supplementary Table 2 — corrections needed to reach APP >= 90% for the candidate
backbone optical-flow algorithms (Pinned / Neural / Sperm / Freely).

Each cell is the median over tracks of the optimal number of corrections
(``optimal_k``) needed to reach APP >= 0.90, read from the per-algorithm benchmark
outputs in ``data/<algo>.json``. The average is the round-half-up mean over the
four datasets (only when all four are present); algorithms are ranked by their
mean over the available datasets. Written to
``results/corrections_to_app90.md``.

    python reproduce_supptable2.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DS = ["pinned", "neural", "sperm", "freely"]
TGT = "0.9"


def cell(data, ds):
    tracks = data.get("datasets", {}).get(ds)
    if not tracks:
        return None
    ks = [t[TGT]["optimal_k"] for t in tracks.values() if TGT in t]
    return int(np.median(ks)) if ks else None


def avg_roundhalfup(cells):
    present = [c for c in cells if c is not None]
    return int(np.mean(present) + 0.5) if len(present) == len(DS) else None


rows = {}
for fn in sorted(os.listdir(DATA)):
    if not fn.endswith(".json") or fn == "errors.json":
        continue
    data = json.load(open(os.path.join(DATA, fn)))
    cells = [cell(data, ds) for ds in DS]
    rows[fn[:-5]] = cells + [avg_roundhalfup(cells)]

order = sorted(rows, key=lambda a: (np.mean([c for c in rows[a][:4] if c is not None])
                                    if any(c is not None for c in rows[a][:4]) else 9999))


def fmt(v):
    return "—" if v is None else str(v)


md = ["# Corrections to reach APP \u2265 90%", "",
      "| Rank | Algorithm | Pinned | Neural | Sperm | Freely | Avg |",
      "|---:|---|---:|---:|---:|---:|---:|"]
print(f"{'#':>2}  {'algorithm':24s}{'pin':>5}{'neu':>5}{'spm':>5}{'fre':>5}{'avg':>5}")
for i, a in enumerate(order, 1):
    r = rows[a]
    print(f"{i:>2}  {a:24s}{fmt(r[0]):>5}{fmt(r[1]):>5}{fmt(r[2]):>5}{fmt(r[3]):>5}{fmt(r[4]):>5}")
    md.append(f"| {i} | {a} | {fmt(r[0])} | {fmt(r[1])} | {fmt(r[2])} | "
              f"{fmt(r[3])} | {fmt(r[4])} |")

os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
open(os.path.join(HERE, "results", "corrections_to_app90.md"), "w").write(
    "\n".join(md) + "\n")
print(f"\n{len(rows)} algorithms -> results/corrections_to_app90.md")
