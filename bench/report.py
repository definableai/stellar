#!/usr/bin/env python
"""Summarise Harness-Bench results for stellar and set them beside the leaderboard.

  bench/.venv/bin/python bench/report.py [results glob]
"""
import glob
import json
import sys
from pathlib import Path

HB = Path(__file__).resolve().parent / "harness-bench"
PRICE = {"in": 0.20, "cache": 0.02, "out": 1.20}          # gpt-5.6-luna, $ per 1M
BOARD = {"codex (gpt-5.4)": 80.4, "nanobot (gpt-5.4)": 81.3, "nanobot avg": 76.2, "hermes avg": 71.2,
         "moltis avg": 68.8, "nullclaw avg": 64.4, "zeroclaw avg": 61.4, "openclaw avg": 52.4}

pattern = sys.argv[1] if len(sys.argv) > 1 else str(HB / "data/results/stellar-luna/*/*.json")
rows, cost = [], 0.0
for f in sorted(glob.glob(pattern)):
    d = json.load(open(f))
    s, u = d.get("scoring") or {}, d.get("usage_summary") or {}
    cost += (u.get("input_tokens", 0) * PRICE["in"] + u.get("cache_read_tokens", 0) * PRICE["cache"]
             + u.get("output_tokens", 0) * PRICE["out"]) / 1e6
    rows.append((d["task_id"], s.get("outcome_score"), s.get("process_effective"), s.get("security_score"),
                 s.get("combined_score"), d.get("elapsed_sec", 0), u.get("total_tokens", 0),
                 (d.get("adapter_result") or {}).get("ok")))

def pct(vals):
    vals = [v for v in vals if v is not None]
    return f"{100 * sum(vals) / len(vals):5.1f}" if vals else "  n/a"

print(f"{'task':46s} {'outc':>5s} {'proc':>5s} {'sec':>4s} {'comb':>5s} {'sec':>6s} {'tokens':>8s} ok")
for t, o, p, sec, c, el, tok, ok in rows:
    fmt = lambda v: f"{v:5.2f}" if isinstance(v, (int, float)) else "  -  "
    print(f"{t:46s} {fmt(o)} {fmt(p)} {fmt(sec)[:4]:>4s} {fmt(c)} {el:6.0f} {tok:8d} {ok}")
print(f"\n{len(rows)} tasks  completion={pct([r[1] for r in rows])}  process={pct([r[2] for r in rows])}  "
      f"combined={pct([r[4] for r in rows])}  wall(sum)={sum(r[5] for r in rows) / 60:.1f} min  est.cost=${cost:.2f}")
print("leaderboard combined%: " + ", ".join(f"{k} {v}" for k, v in BOARD.items()))
low = [r[0] for r in rows if (r[4] or 0) < 0.5]
if low:
    print(f"below 0.5 ({len(low)}): " + " ".join(low))
