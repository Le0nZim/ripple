#!/usr/bin/env python3
"""
Table 2 — accuracy and practical cost for RIPPLE and baseline trackers on the
Neural *Clytia* dataset.

The APP and manual-annotation columns are read from the benchmark output
``data/benchmark_results.json`` (produced by the full baseline-evaluation
pipeline ``benchmark.py``; see the README). The elapsed-time, computation-time
and hyper-parameter columns are logged measurements.

    python reproduce_table2.py        # prints the table and writes results/table2.md
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "data", "benchmark_results.json")))
APP = DATA["aggregate_app"]
ANNOT = DATA["annotations"]

# Logged time / hyper-parameter measurements, and the JSON key for each method.
ROWS = [
    # label, json_key, elapsed, compute, hyper_params
    ("RIPPLE",    "Ripple",    "15 min",    "3.7 s",     0),
    ("TrackMate", "TrackMate", "33.2 min",  "3.7 s",     357),
    ("LocoTrack", "LocoTrack", "2.3 s",     "2.3 s",     0),
    ("SLEAP",     "SLEAP",     "156.6 min", "141.7 min", 0),
    ("SLEAP-op",  "SLEAP-op",  "97.2 min",  "82.2 min",  0),
]

header = ("| Method | APP (%) | Manual annotations | Total elapsed time | "
          "Computation time | Hyper-parameter combinations |")
sep = "|---|---:|---:|---|---|---:|"
md = [header, sep]
print(f"{'Method':<10}{'APP (%)':>9}{'annot':>7}{'elapsed':>11}{'compute':>11}"
      f"{'hyper':>7}")
for label, key, elapsed, compute, hyper in ROWS:
    app = APP[key]["average"]
    annot = ANNOT[key]
    print(f"{label:<10}{app:>9.2f}{annot:>7}{elapsed:>11}{compute:>11}{hyper:>7}")
    md.append(f"| {label} | {app:.2f} | {annot} | {elapsed} | {compute} | {hyper} |")

os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
out = os.path.join(HERE, "results", "table2.md")
with open(out, "w") as f:
    f.write("# Table 2 — RIPPLE vs. baseline trackers (Neural *Clytia*)\n\n"
            + "\n".join(md) + "\n")
print(f"\nwrote {os.path.relpath(out, HERE)}")
