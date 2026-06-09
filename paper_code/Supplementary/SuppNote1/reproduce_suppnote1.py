#!/usr/bin/env python3
"""
Supplementary Note 1 — manual-intervention cost of converting TrackMate output
into the tracks of interest, compared with RIPPLE.

For each matched ground-truth track, the analysis replays, frame by frame, the
three manual interventions a user would need starting from TrackMate's output:
  * initial_pick : choose one TrackMate trajectory at the first GT frame,
  * relink       : the nearest detection within tolerance jumps to a different
                   TrackMate track -> merge / re-link,
  * manual_click : no detection within tolerance -> place the point by hand.
Total interventions = initial_pick + relink + manual_click, compared with the
number of RIPPLE corrections (anchors).

Reads ``data/intervention_inputs.json`` (from ``generate_data.py``) and writes a
summary to ``results/intervention_cost.md``.

    python reproduce_suppnote1.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "data", "intervention_inputs.json")))
ORDER = ["Neural Activity", "Pinned Down", "Freely", "Sperm", "Homeostasis"]


def analyse(track, dets_by_frame, n_tm_frames, tol):
    visible = {f: (x, y) for f, x, y in track["visible"]}
    n_total = track["n_total_frames"]
    n_avail = min(n_total, n_tm_frames)
    initial_pick = relink = manual = gt_visible = 0
    followed_tid = None
    started = False
    for f in range(n_avail):
        if f not in visible:
            continue
        gt_visible += 1
        gx, gy = visible[f]
        best_tid, best_d = None, tol
        for tid, tx, ty in dets_by_frame.get(f, ()):
            d = ((tx - gx) ** 2 + (ty - gy) ** 2) ** 0.5
            if d < best_d:
                best_d, best_tid = d, tid
        if best_tid is None:
            manual += 1
            followed_tid = None
            continue
        if not started:
            initial_pick += 1
            started = True
            followed_tid = best_tid
            continue
        if best_tid != followed_tid:
            relink += 1
            followed_tid = best_tid
    for f in range(n_avail, n_total):
        if f in visible:
            gt_visible += 1
            manual += 1
    return gt_visible, initial_pick + relink + manual


md = ["# TrackMate vs. RIPPLE manual-intervention cost", "",
      "| Dataset | GT frames | TrackMate | RIPPLE | TM / RIPPLE |",
      "|---|---:|---:|---:|---:|"]
print(f"{'dataset':<16}{'gt_frm':>8}{'TrackMate':>10}{'RIPPLE':>8}{'TM/RIP':>8}")
print("-" * 50)
tot_gt = tot_tm = tot_rip = 0
for name in ORDER:
    d = DATA[name]
    dets_by_frame = {}
    for f, tid, tx, ty in d["detections"]:
        dets_by_frame.setdefault(f, []).append((tid, tx, ty))
    gt_sum = tm_sum = rip_sum = 0
    for tr in d["tracks"]:
        gv, iv = analyse(tr, dets_by_frame, d["n_tm_frames"], d["tol"])
        gt_sum += gv; tm_sum += iv; rip_sum += tr["ripple_anchors"]
    ratio = tm_sum / rip_sum if rip_sum else float("inf")
    print(f"{name:<16}{gt_sum:>8}{tm_sum:>10}{rip_sum:>8}{ratio:>7.1f}x")
    md.append(f"| {name} | {gt_sum} | {tm_sum} | {rip_sum} | {ratio:.1f}× |")
    tot_gt += gt_sum; tot_tm += tm_sum; tot_rip += rip_sum

ratio = tot_tm / tot_rip
print("-" * 50)
print(f"{'ALL':<16}{tot_gt:>8}{tot_tm:>10}{tot_rip:>8}{ratio:>7.1f}x")
md.append(f"| **All** | {tot_gt} | {tot_tm} | {tot_rip} | **{ratio:.1f}×** |")
md += ["", f"Overall, TrackMate requires {ratio:.1f}× more manual interventions "
       f"than RIPPLE to reconstruct the same tracks."]

os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
open(os.path.join(HERE, "results", "intervention_cost.md"), "w").write(
    "\n".join(md) + "\n")
print(f"\nOverall TrackMate / RIPPLE intervention ratio: {ratio:.1f}x  "
      f"-> results/intervention_cost.md")
