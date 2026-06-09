#!/usr/bin/env python3
"""
Supplementary Table — candidate backbone optical-flow algorithms.

The table is the algorithm *registry* of the benchmark in ``../SuppTable2``. It is
parsed straight from that benchmark's source (`benchmark_optical_flows.py`) so the
list always matches the code that produced the results, and written to
``results/algorithm_list.md``.

    python reproduce_supptable1.py
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "SuppTable2", "benchmark_optical_flows.py")

DEFAULT = "dis_medium"                       # backbone used by RIPPLE
ERRORED = {"ptl_matchflow", "ptl_separableflow", "ptl_streamflow"}  # failed to run
PKG_ORDER = ["opencv", "opencv_contrib", "scikit-image", "pyoptflow",
             "torchvision", "ptlflow"]


def parse_registry(text):
    reg = []
    for name, pkg in re.findall(
            r'ALGORITHMS\.append\(\(\s*"([a-z0-9_]+)",\s*"([a-z_\-]+)"', text):
        reg.append((name, pkg))
    m = re.search(r"PTLFLOW_MODELS\s*=\s*\[(.*?)\]", text, re.S)
    for pmodel in re.findall(r'\(\s*"([a-z0-9_]+)"', m.group(1)):
        reg.append(("ptl_" + pmodel, "ptlflow"))
    return reg


def main():
    reg = parse_registry(open(SRC).read())
    by_pkg = {}
    for name, pkg in reg:
        by_pkg.setdefault(pkg, []).append(name)

    out = ["# Candidate backbone optical-flow algorithms", "",
           "| Package | Algorithm | Notes |", "|---|---|---|"]
    for pkg in PKG_ORDER:
        for name in by_pkg.get(pkg, []):
            note = ("default in RIPPLE" if name == DEFAULT else
                    "no result (failed to run)" if name in ERRORED else "")
            out.append(f"| {pkg} | {name} | {note} |")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    open(os.path.join(HERE, "results", "algorithm_list.md"), "w").write(
        "\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\n{len(reg)} algorithms across {len(by_pkg)} packages.")


if __name__ == "__main__":
    main()
