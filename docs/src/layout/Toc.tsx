import { List } from "lucide-react";
import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";

type Heading = { id: string; text: string; depth: number };

const TOP = 88;   // where the page begins under the topbar: prose.css scrolls headings to it

const same = (a: Heading[], b: Heading[]) => a.length === b.length && a.every((h, i) => h.id === b[i].id && h.text === b[i].text);

export default function Toc() {
  const { pathname, hash } = useLocation();
  const [headings, setHeadings] = useState<Heading[]>([]);
  const [active, setActive] = useState("");

  useEffect(() => {
    const main = document.getElementById("content");
    if (!main) return;
    const read = () => {
      const found = [...main.querySelectorAll<HTMLElement>("article h2, article h3")]
        .filter((h) => h.id)
        .map((h) => ({ id: h.id, text: h.textContent ?? "", depth: h.tagName === "H3" ? 3 : 2 }));
      // The same headings must stay the same array: the observer below hangs off it.
      setHeadings((now) => (same(now, found) ? now : found));
    };
    read();
    // the page is lazy: read again when it mounts.
    const watch = new MutationObserver(read);
    watch.observe(main, { childList: true, subtree: true });
    return () => watch.disconnect();
  }, [pathname]);

  // Scrollspy: the topmost heading in the band under the topbar is the one being read.
  useEffect(() => {
    if (!headings.length) return;
    const band = new Set<string>();
    const top = (id: string) => document.getElementById(id)?.getBoundingClientRect().top ?? Infinity;
    const watch = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) entry.isIntersecting ? band.add(entry.target.id) : band.delete(entry.target.id);
        // in the band, else the last one that has left it over the top, else the first.
        setActive((headings.find((h) => band.has(h.id)) ?? headings.filter((h) => top(h.id) <= TOP).at(-1) ?? headings[0]).id);
      },
      { rootMargin: `-${TOP}px 0px -70% 0px` },
    );
    for (const heading of headings) {
      const element = document.getElementById(heading.id);
      if (element) watch.observe(element);
    }
    return () => watch.disconnect();
  }, [headings]);

  useEffect(() => { if (hash) setActive(decodeURIComponent(hash.slice(1))); }, [hash]);

  if (!headings.length) return null;
  return (
    <aside className="sticky top-16 hidden max-h-[calc(100vh-4rem)] self-start overflow-y-auto pt-4 pb-10 [scrollbar-width:thin] xl:block">
      <div className="flex items-center gap-2 text-sm font-medium text-gray-600 dark:text-gray-300">
        <List size={16} className="shrink-0" />
        On this page
      </div>
      <nav className="mt-2">
        {headings.map((heading) => (
          <Item key={heading.id} heading={heading} active={heading.id === active} />
        ))}
      </nav>
    </aside>
  );
}

/** One line of the table. */
function Item({ heading, active }: { heading: Heading; active: boolean }) {
  return (
    <a
      href={`#${heading.id}`}
      className={`relative block py-1 text-sm leading-6 ${heading.depth === 3 ? "pl-8" : "pl-4"} ${
        active ? "text-primary dark:text-primary-light" : "text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-200"
      }`}
    >
      {active && <span className="absolute top-1/2 left-1.5 h-1 w-1 -translate-y-1/2 rounded-full bg-current" />}
      {heading.text}
    </a>
  );
}
