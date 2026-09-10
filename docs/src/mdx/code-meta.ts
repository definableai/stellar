/** The fence's meta string, on both sides of the build.
 *
 * Grammar (Mintlify's spellings), words in any order:
 * `title="main.py"` the header title, quoted so it may hold spaces; `{1,3-5}`
 * highlighted rows, read by rehype-pretty-code, not here; `lines` `wrap`
 * `expandable` `nocopy` `run` `output` the flags; whatever is left over is the title too,
 * which is how ```python main.py names a block.
 */

export type Meta = { title: string; lines: boolean; wrap: boolean; expandable: boolean; nocopy: boolean; run: boolean };

// `run`: check.py --run executes the fence; `output`: the text fence check.py compares it with.
const FLAGS = ["lines", "wrap", "expandable", "nocopy", "run", "output"];

export function parseMeta(meta = ""): Meta {
  const words: string[] = meta.match(/title="[^"]*"|\S+/g) ?? [];
  const quoted = words.find((word) => word.startsWith('title="'));
  const left = words.filter((word) => word !== quoted && !FLAGS.includes(word) && !word.startsWith("{"));
  return {
    title: quoted ? quoted.slice(7, -1) : left.join(" "),
    lines: words.includes("lines"),
    wrap: words.includes("wrap"),
    expandable: words.includes("expandable"),
    nocopy: words.includes("nocopy"),
    run: words.includes("run"),
  };
}

/** Only the fields the walk below touches; hast is not a dependency here. */
type Node = { tagName?: string; properties?: Record<string, unknown>; data?: { meta?: string }; children?: Node[] };

/**
 * A rehype plugin, run after rehype-pretty-code: that one rewrites the `pre`
 * into a `figure` and leaves the meta on `code.data`, where JSX cannot see it.
 * Put the meta back on the `pre` as a prop, and drop the figure — the frame is
 * CodeBlock's to draw.
 */
export function codeMeta() {
  return (tree: unknown) => walk(tree as Node);
}

function walk(node: Node) {
  const children = node.children ?? [];
  for (const [i, child] of children.entries()) {
    const pre = "data-rehype-pretty-code-figure" in (child.properties ?? {})
      ? child.children?.find((c) => c.tagName === "pre")
      : undefined;
    if (!pre) {
      walk(child);
      continue;
    }
    const meta = pre.children?.[0]?.data?.meta;
    if (meta) pre.properties = { ...pre.properties, "data-meta": meta };
    children[i] = pre;
  }
}
