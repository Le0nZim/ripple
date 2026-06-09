# Supplementary Table 1 — candidate backbone optical-flow algorithms

The 51 optical-flow algorithms considered as the motion backbone for RIPPLE.

## Pipeline

```
../SuppTable2/benchmark_optical_flows.py  ──(reproduce_supptable1.py)──►  results/algorithm_list.md
```

```bash
python reproduce_supptable1.py     # writes results/algorithm_list.md
```

The list is the algorithm registry parsed directly from the benchmark source in
[`../SuppTable2`](../SuppTable2/benchmark_optical_flows.py), so it always matches
the code that produced the Supplementary Table 2 results. `dis_medium` is the
backbone RIPPLE uses; three `ptlflow` models failed to run and are noted. Requires
only the standard library.
