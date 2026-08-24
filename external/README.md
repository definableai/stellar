# external — the agent's workspace

Adapters the agent writes for itself live here, loaded by the kernel
(`core/kernel.py`). Ships empty of code: whatever appears below was written
by an agent, not by us.

- Each adapter is a `*.py` defining `setup(ctx)` — the same shape as the
  `internal/` adapters. Subfolders are fine (mounted recursively, sorted by
  path, so `tool/x.py` mounts as `tool/x`); prefix a file or folder with `_`
  to disable it.
- The path is configuration, never a core default: `boot(agent, "external")`
  mounts the kernel plus everything here (see `examples/cc/cc.py`).
- Keep it under git — then every self-change is a reviewable diff.
