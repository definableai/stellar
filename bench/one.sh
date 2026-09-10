#!/usr/bin/env bash
# One task: run it, print one summary line. Env comes from run.sh.
t=$1
cd "$HB" && "$B/.venv/bin/python" -m harnessbench.cli run-task --task "$t" --harness stellar-luna --mode live >"$B/logs/$t.log" 2>&1
rc=$?
f=$(ls -t "$HB"/data/results/stellar-luna/*/"$t".json 2>/dev/null | head -1)
[ -n "$f" ] && "$B/.venv/bin/python" - "$t" "$rc" "$f" <<'PY' || echo "$t rc=$rc no-result"
import json, sys
t, rc, f = sys.argv[1:]
d = json.load(open(f)); s = d.get("scoring") or {}
print(f"{t} rc={rc} combined={s.get('combined_score')} outcome={s.get('outcome_score')} process={s.get('process_effective')} elapsed={d.get('elapsed_sec')}s")
PY
