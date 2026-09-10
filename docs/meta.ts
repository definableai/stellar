/** `virtual:meta`: every page's frontmatter, read once at build.
 *
 * The sidebar, the breadcrumb and the pagination all name pages the reader has
 * not opened. Reading it here keeps the MDX itself lazy — the source ships only
 * when search or "Copy page" asks for it.
 */
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { Plugin } from "vite";

const ID = "virtual:meta";
const RESOLVED = "\0" + ID;
const FRONTMATTER = /^---\r?\n([\s\S]*?)\r?\n---/;

/** The `key: value` lines between the dashes, the way check.py reads them. */
function frontmatter(text: string) {
  const block = FRONTMATTER.exec(text)?.[1] ?? "";
  const front: Record<string, string> = {};
  for (const line of block.split("\n")) {
    const colon = line.indexOf(":");
    if (colon > 0) front[line.slice(0, colon).trim()] = line.slice(colon + 1).trim().replace(/^["']|["']$/g, "");
  }
  return front;
}

export default function meta(): Plugin {
  let dir = "";
  return {
    name: "docs-meta",
    configResolved: (config) => { dir = join(config.root, "content"); },
    resolveId: (id) => (id === ID ? RESOLVED : undefined),
    load(id) {
      if (id !== RESOLVED) return;
      const pages: Record<string, Record<string, string>> = {};
      for (const file of readdirSync(dir, { recursive: true }) as string[]) {
        if (!file.endsWith(".mdx")) continue;
        this.addWatchFile(join(dir, file));       // a title edit reloads in dev
        pages[file.slice(0, -4)] = frontmatter(readFileSync(join(dir, file), "utf8"));
      }
      return `export default ${JSON.stringify(pages)}`;
    },
  };
}
