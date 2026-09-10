/** The MDX under content/: one lazy module per page, and the source it was cut from. */
import { idOf } from "./site";

const file = (path: string) => `../content/${idOf(path)}.mdx`;

const modules = import.meta.glob("../content/**/*.mdx");
// Eager: the sidebar needs every page's frontmatter title before the page is
// opened. Ceiling: all MDX source ships in the entry chunk (~kBs a page).
const sources = import.meta.glob("../content/**/*.mdx", { query: "?raw", import: "default", eager: true }) as Record<string, string>;

/** The lazy module for a page, or undefined when there is no such file. */
export const load = (path: string) => modules[file(path)];

/** The raw MDX of a page, frontmatter included. */
export const raw = (path: string): string | undefined => sources[file(path)];

const FRONTMATTER = /^---\r?\n([\s\S]*?)\r?\n---/;

/** A page's frontmatter, parsed the way check.py parses it. */
export function meta(path: string): Record<string, string> {
  const block = FRONTMATTER.exec(raw(path) ?? "")?.[1] ?? "";
  const front: Record<string, string> = {};
  for (const line of block.split("\n")) {
    const colon = line.indexOf(":");
    if (colon > 0) front[line.slice(0, colon).trim()] = line.slice(colon + 1).trim().replace(/^["']|["']$/g, "");
  }
  return front;
}
