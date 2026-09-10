/** A ```mermaid fence is a picture, not code. */

/** Only the fields the walk below touches; mdast is not a dependency here. */
type Node = { type: string; lang?: string; value?: string; children?: Node[] };

/**
 * A remark plugin, so it lands before the rehype stage and rehype-pretty-code
 * never sees the fence: a ```mermaid block becomes `<Mermaid chart="…" />`,
 * a name mdx/components.ts maps like every other component.
 */
export function mermaid() {
  return (tree: unknown) => walk(tree as Node);
}

function walk(node: Node) {
  const children = node.children ?? [];
  for (const [i, child] of children.entries()) {
    if (child.type !== "code" || child.lang !== "mermaid") {
      walk(child);
      continue;
    }
    children[i] = {
      type: "mdxJsxFlowElement",
      name: "Mermaid",
      attributes: [{ type: "mdxJsxAttribute", name: "chart", value: child.value ?? "" }],
      children: [],
    } as unknown as Node;
  }
}
