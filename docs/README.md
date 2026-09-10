# The docs

The writing and the machine that renders it, in one folder. Pages are MDX
under `content/`, `docs.json` is the table of contents (Mintlify's schema),
and `src/` is a small Vite + React app that reads both. Static build, no
server.

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # tsc + vite, into dist/ (postbuild copies index.html to 404.html)
npm run check    # python3 check.py --run — nav, links, icons, fences; every `run` example runs
npm run check    # python3 check.py — nav, links, frontmatter, python fences
npm run check:diagrams   # every ```mermaid fence parses
```

## Adding a page

1. Write `content/<path>.mdx`. The file path is the URL: `content/tools/parallel.mdx`
   serves `/tools/parallel`, and `content/introduction.mdx` also serves `/`.
2. List it in `docs.json` under the group it belongs to, without the extension
   (`"tools/parallel"`). A page not listed there needs `hidden: true`.
3. Give it frontmatter:

```mdx
---
title: Parallel tools          # the h1 and the sidebar label
description: One barrier, ...  # the line under the h1
sidebarTitle: Parallel         # optional, a shorter sidebar label
---
```

A code example the reader can run is a fence tagged `run`, followed by a
`text output` fence with exactly what it prints. `check.py --run` executes
it from the repo root with no API key and compares. The block gets a
"Tested" chip on the page. An example that needs a key is not tagged `run`.

No `# H1` in the body — the title is the h1. Use `##` for sections and `###`
inside them. Links are absolute site paths (`[the loop](/concepts/loop)`);
`check.py` fails a relative one and one that points at no page.

Every Mintlify component works by name, with no import line — `Note`, `Card`,
`Tabs`, `Steps`, `ParamField` and the rest. A `mermaid` fence is drawn as a
diagram, themed for both modes. `/kitchen-sink` shows them all.

The look — tokens, measurements, the component table — is specified in
`tasks/docs-site/00-plan.md`.
