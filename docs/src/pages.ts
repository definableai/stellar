/** The MDX under content/: one lazy module per page, and the source it was cut from. */
import front from "virtual:meta";
import { idOf } from "./site";

const file = (path: string) => `../content/${idOf(path)}.mdx`;

const modules = import.meta.glob("../content/**/*.mdx");
const sources = import.meta.glob("../content/**/*.mdx", { query: "?raw", import: "default" }) as Record<string, () => Promise<string>>;

/** The lazy module for a page, or undefined when there is no such file. */
export const load = (path: string) => modules[file(path)];

/** Loads the raw MDX of a page, frontmatter included — search and "Copy page" ask, no one else. */
export const raw = (path: string) => sources[file(path)];

/** A page's frontmatter, read at build: naming a page costs no source. */
export const meta = (path: string): Record<string, string> => front[idOf(path)] ?? {};
