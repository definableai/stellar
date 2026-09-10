/** The index and the scorer: every page cut into sections at its headings.
 *
 * Built the first time ⌘K opens, from the raw MDX. Ceiling: a linear scan over
 * every section on every keystroke — fine to a few hundred pages.
 */
import GithubSlugger from "github-slugger";
import { meta, raw } from "./pages";
import { find, flatten, tabs } from "./site";

export type Page = { path: string; title: string; group: string; tab: string };
export type Section = { page: Page; heading: string; anchor: string; text: string };

const FRONTMATTER = /^---\r?\n[\s\S]*?\r?\n---/;
const FENCE = /^[ \t]*```[\s\S]*?^[ \t]*```/gm;
const MDX_LINES = /^(import|export) .*$/gm;
const COMMENT = /\{\/\*[\s\S]*?\*\/\}/g;
const HEADING = /^#{1,6} +(.*)$/gm;               // split keeps the capture: text, heading, text, …
const MARKUP = /<[^>]*>/g;                        // a JSX tag; its inner text stays

/** Prose as a reader sees it: no markup, no markdown punctuation, one space between words. */
const plain = (text: string) =>
  text.replace(MARKUP, " ").replace(/\[([^\]]*)\]\([^)]*\)/g, "$1").replace(/[`*_>|#]/g, "").replace(/\s+/g, " ").trim();

/** Every listed page, in reading order — the order ties fall back on. */
const listed = tabs.flatMap(flatten);

let building: Promise<Section[]> | undefined;

/** The sections of every page, built once. */
export const index = () => (building ??= Promise.all(listed.map(sections)).then((all) => all.flat()));

async function sections(path: string): Promise<Section[]> {
  const source = raw(path);
  const here = find(path);
  if (!source || !here) return [];
  const front = meta(path);
  const page = { path, title: front.title ?? path, group: here.group, tab: here.tab };
  const body = (await source()).replace(FRONTMATTER, "").replace(FENCE, "").replace(MDX_LINES, "").replace(COMMENT, "");
  const [lead, ...runs] = body.split(HEADING);
  const slugger = new GithubSlugger();
  const out: Section[] = [{ page, heading: "", anchor: "", text: `${front.description ?? ""} ${plain(lead)}`.trim() }];
  for (let i = 0; i < runs.length; i += 2) {
    const heading = plain(runs[i]);
    out.push({ page, heading, anchor: slugger.slug(heading), text: plain(runs[i + 1] ?? "") });
  }
  return out;
}

/** Up to 12 sections: a title match beats a heading match beats a body match,
 *  each counted by how many of the query's words land there; ties by reading order. */
export function search(query: string, index: Section[]): Section[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length) return [];
  const hits: { section: Section; score: number }[] = [];
  for (const section of index) {
    const title = section.page.title.toLowerCase();
    const heading = section.heading.toLowerCase();
    const text = section.text.toLowerCase();
    let score = 0;
    for (const word of words) {
      if (title.includes(word)) score += 10000;
      else if (heading.includes(word)) score += 100;
      else if (text.includes(word)) score += 1;
      else { score = 0; break; }                  // every word has to land somewhere
    }
    if (score) hits.push({ section, score });
  }
  return hits.sort((a, b) => b.score - a.score).slice(0, 12).map((hit) => hit.section);
}
