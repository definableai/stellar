#!/usr/bin/env bash
# Run Harness-Bench tasks against stellar, in parallel, under one wall clock.
#   bench/run.sh                 all 106 tasks, longest timeout first
#   bench/run.sh 001-file 002-exec
#   J=8 WALL=1800 JUDGE=gpt-5.6-luna bench/run.sh ...
set -euo pipefail
S=$(cd "$(dirname "$0")/.." && pwd); B=$S/bench; HB=$B/harness-bench
set -a; . "$B/.env"; set +a
: "${OPENAI_API_KEY:?put OPENAI_API_KEY=... in bench/.env}"
export B HB PYTHONPATH="$S:$HB/src" PATH="$B/.venv/bin:$PATH"   # oracles and hooks call python3/pytest
export HARNESSBENCH_HARNESS_CONFIG="$B/harness.yaml"
export RUBRIC_API_KEY="$OPENAI_API_KEY" RUBRIC_BASE_URL=https://api.openai.com/v1
export RUBRIC_MODEL="${JUDGE:-gpt-5.6-luna}" RUBRIC_VISION_MODEL="${JUDGE:-gpt-5.6-luna}"
export STELLAR_TASKS_DIR="$HB/tasks"
export STELLAR_DEADLINE=$(( $(date +%s) + ${WALL:-1800} ))   # every task ends by here
J=${J:-8}
if [ $# -eq 0 ]; then
  set -- $(cd "$HB/tasks" && for t in */task.yaml; do
    printf '%s %s\n' "$(awk '/^timeout_sec/{print $2}' "$t")" "${t%/task.yaml}"; done | sort -rn | awk '{print $2}')
fi
mkdir -p "$B/logs"
echo "tasks=$# jobs=$J wall=${WALL:-1800}s judge=${JUDGE:-gpt-5.6-luna}"
printf '%s\n' "$@" | xargs -P "$J" -n 1 "$B/one.sh"
