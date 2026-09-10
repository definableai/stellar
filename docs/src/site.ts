/** docs.json read once: the tabs, the reading order, and where a page sits. */
import docs from "../docs.json";

export type Group = { group: string; pages: (string | Group)[] };
export type Tab = { tab: string; groups: Group[] };

export const { name, colors, appearance, navbar } = docs;
export const tabs = docs.navigation.tabs as unknown as Tab[];

/** The page id behind a URL path: "/concepts/loop" → "concepts/loop", "/" → "introduction". */
export const idOf = (path: string) => path.replace(/^\/+|\/+$/g, "") || "introduction";

/** Every page under a tab or a group, in reading order, with the group it sits in. */
function walk(node: Tab | Group): [page: string, group: string][] {
  if ("groups" in node) return node.groups.flatMap(walk);
  return node.pages.flatMap((p) => (typeof p === "string" ? [[p, node.group] as [string, string]] : walk(p)));
}

export const flatten = (node: Tab | Group) => walk(node).map(([page]) => page);

/** The tab, group and neighbours of a page, or undefined when docs.json does not list it. */
export function find(path: string) {
  const page = idOf(path);
  for (const tab of tabs) {
    const rows = walk(tab);
    const at = rows.findIndex(([p]) => p === page);
    if (at >= 0) return { tab: tab.tab, group: rows[at][1], page, prev: rows[at - 1]?.[0], next: rows[at + 1]?.[0] };
  }
}
