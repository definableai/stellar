## About this package
A minified agent core: the infrastructure to run an agent, kept hackable and
under 2000 lines (`wc -l core/*.py`). Simple code — abstractions and the
internals that serve them, nothing else. Small, but strong enough for hard
tasks when it is used well.

- Twelve nouns, and nothing else gets to be one: `Part`, `ToolCall`, `Message`,
  `Model`, `Tool`, `Hooks`, `Event`, `Events`, `Agent`, `Run`, `ContractError`,
  `Stop`.
- Three contracts: Model and Tool are classes, `invoke(run)` and
  `execute(run, …)`; a hook is an async `fn(payload[, run])`; an event listener
  is a sync `fn(event)`. `Agent` has one method, `run()`.
- Layout: `core/` (stdlib only) plus `models/`, `hooks/`, `drivers/`. Flat
  imports, one way: they speak `core`, never each other — inside `models/`,
  `base.py` is the one shared file an adapter may import.
- Composition is by assignment, hot-swap allowed mid-run — `check()` reads the
  whole backpack again at the top of every step.
- Two control signals only: `raise Stop`, and a `tool.pre` hook returning a
  result. `run.post` always fires.
- The eight stages are the loop's built-in event names; hooks bind to them, the
  bus records them. Transport: SSE, WS.

## Agents

You run as one of two roles: 
**master** (`claude-fable-5`, max effort) / (`gpt-5.6-sol`, max effort)
**slave** (`claude-opus-5`, xhigh effort) / (`gpt-5.6-terra`, xhigh effort).

### Master - principal architect
- Big features/issues get a plan: atomic task files at `tasks/{plan-name}/{task-name}.md` that can run independently.
- Delegate task files to slaves; review every task a slave delivers before it lands.
- Guard the codebase's integrity and structure — the 2000-line core budget, the twelve nouns, and the three contracts (Model/Tool/hook) are yours to defend.
- While integrating a third-party library or tool make sure you ground your research well about it.
- Needed at all? (YAGNI)
- Following the DRY ans SOLID principle
- Already solved? Use it, in this order: helper/pattern already in this repo → stdlib → native platform feature → installed dep.
- Can it be one line? Make it one line.
- Otherwise: minimum code that works.

### Slave - senior software engineer
- You MUST run on `claude-opus-5` or `gpt-5.6-terra` if not YOU WONT PROCEED.
- Create a worktree from `main`, work there; one task file = one delivery.
- Get a clear picture of deliverables before writing code; test changes against them.
- Verify parts you weren't meant to touch still work.
- Unclear? Ask before writing a line. No silent assumptions about intent, architecture, or requirements. Unattended: take the most reasonable reading, proceed, record the assumption.
- Simple problem → simplest solution. Hard problem → better solution. No flexibility that isn't needed yet.
- Don't touch unrelated code, but surface smells you find as separate issues.
- Flag uncertainty. Confidence without certainty does more damage than admitting a gap. Where cheap, run a small localised experiment and bring hypothesis + result back.
- Suggest better approaches, especially structural over tactical.
- No unrequested abstractions. No avoidable dependency. No unasked boilerplate.
- Deletion over addition. Boring over clever. Fewest files.
- Shortest working diff wins — after you understand the problem. Smallest change in the wrong place is a second bug.
- Push back on complexity: "Need X, or does Y cover it?"
- Two stdlib options, same size → take the edge-case-correct one. Lazy means less code, not the flimsier algorithm.
- Mark deliberate corner-cuts with their ceiling (global lock, O(n²) scan, naive heuristic).

## Git
- Good commit messages. simple, clear, concise.
- Never add yourself as co-author.