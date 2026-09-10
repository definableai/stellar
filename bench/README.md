# stellar on Harness-Bench

The wiring that puts stellar on [Harness-Bench](https://github.com/Qihoo360/harness-bench),
and the record of every pass. Nothing in here is imported by `core/`.

| file | what |
| --- | --- |
| `stellar_cli.py` | the bench's `generic_cli` side: builds an `Agent` on `core` with five tools (`bash`, `read_file`, `write_file`, `edit_file`, `view_image`), `Steps`, `Budget` and a `Log`; keeps a session per task under the sandbox; writes the judge trace in the proxy's format |
| `responses.py` | the Responses API provider the passes ran on: `previous_response_id` chaining, reasoning on |
| `harness.yaml` | the bench entry `stellar-luna`; the two paths in it are absolute, edit them for your checkout |
| `run.sh`, `one.sh` | the whole suite (or the tasks you name) in parallel under one wall clock; one summary line per task |
| `report.py` | the per-task table, the totals, and the leaderboard beside them |
| `harness-bench.patch` | what was changed in the clone (see below) |
| `logs/`, `logs-pass*/` | one `<task>.log` per task, `full-run.txt` as the tasks finished, `report-pass*.txt` as scored. `logs/` is the latest pass |

Not committed: `harness-bench/`, a clone of the bench (408 MB, its own repo);
`.venv/`; `.env` with the key.

## Setup

```bash
git clone https://github.com/Qihoo360/harness-bench bench/harness-bench
git -C bench/harness-bench checkout 1025086
git -C bench/harness-bench apply ../harness-bench.patch
uv venv bench/.venv
uv pip install --python bench/.venv/bin/python httpx pyyaml python-docx openpyxl pillow pypdf pytest
echo 'OPENAI_API_KEY=sk-...' > bench/.env
```

Edit the two absolute paths in `harness.yaml`. Then:

```bash
STELLAR_FAKE=1 HARNESSBENCH_SKIP_PROCESS_GRADE=1 J=1 bench/run.sh 001-file   # plumbing only, no tokens
J=12 WALL=1800 bench/run.sh                                                 # all 106 tasks, 30-minute wall
bench/run.sh 001-file 002-exec                                              # a few
bench/.venv/bin/python bench/report.py                                      # the table
```

Knobs, all environment: `STELLAR_MODEL` (gpt-5.6-luna), `STELLAR_EFFORT` (high),
`STELLAR_REVIEWS` (0: a reviewer pass by a second agent, off), `STELLAR_MAX_STEPS`
(60), `STELLAR_MAX_TOKENS` (1.2M), `STELLAR_CAP_SEC` (1500), `JUDGE` (the rubric
model, gpt-5.6-luna). The model is gpt-5.6-luna throughout; the judge is the
same model.

## The passes

106 tasks each, twelve at a time, judge gpt-5.6-luna. Cost is an estimate from
token counts at list price.

| pass | when | setup | completion | process | combined | cost |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | 2026-09-10 | Chat Completions, no reasoning; twenty runs died on rate limits | 73.3 | 93.4 | 69.1 | $0.29 |
| 2 | 2026-09-10 | retries, checklist-first prompt, a final-check turn | 83.6 | 96.9 | 81.2 | $0.57 |
| 3 | 2026-09-10 | as 2 | 84.4 | 96.6 | 81.6 | $0.60 |
| 4 | 2026-09-10 | Responses API with reasoning, a reviewer agent | 86.4 | 92.8 | 80.1 | $2.27 |
| 5 | 2026-09-11 | a more conservative reviewer | 88.0 | 93.5 | 82.4 | $2.13 |
| 6 | 2026-09-11 | a clipped, bash-only reviewer; stopped after 43 tasks | | | | |
| 7 | 2026-09-11 | reasoning high, no reviewer, the worker checks its own work | 85.5 | 97.4 | **83.3** | $2.41 |

The reviewer traded outcome for process: the judge reads only the first 24K
characters of the trace, and the reviewer's tool output was half of it. Pass 7
is what the README reports. Between passes the same task moves about twelve
points; the mean is steady to about one.

## The patch

Three edits to the clone, applied on top of upstream `1025086`:

- `config/app.yaml`: data directories, and a 1700 s default timeout so every
  task ends inside the 30-minute wall.
- `grading/rubric_llm.py`: `temperature` is left out of the judge call for
  gpt-5.x models, which reject any value but the default. Plumbing, not
  scoring.
- `tasks.py`: an oracle that crashes scores the task 0 instead of killing the
  run. Conservative — it can only lower a score.
