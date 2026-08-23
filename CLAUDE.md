## Abou this packages
This is a minified agent core and its task its to provide agent infra, and idea is to keep this core hackable and under 2000 lines of code. Simple code, just abstractions, and internal functionality. 
But this mini core must stay powerfull, can handle complex tasks if implemented properly
- Layers : LLM, TOOL, HOOK
- Transport : SSE, WS
- Events : step_start, step_delta, step_end
- Tracing : Jsonl, Console

## Before writing code
- Needed at all? (YAGNI)
- Following the DRY ans SOLID principle
- Already solved? Use it, in this order: helper/pattern already in this repo → stdlib → native platform feature → installed dep.
- Can it be one line? Make it one line.
- Otherwise: minimum code that works.

## While writing code
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

## After writing code(Verification)
- Always use a subagent with `claude-opus-4.6` model to do the smoke testing.

## Git
- Good commit messages. simple, clear, concise.
- Never add yourself as co-author.