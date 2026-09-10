import { List } from "lucide-react";
import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";

type Heading = { id: string; text: string; depth: number };

export default function Toc() {
  const { pathname } = useLocation();
  const [headings, setHeadings] = useState<Heading[]>([]);

  useEffect(() => {
    const main = document.getElementById("content");
    if (!main) return;
    const read = () =>
      setHeadings(
        [...main.querySelectorAll<HTMLElement>("article h2, article h3")]
          .filter((h) => h.id)
          .map((h) => ({ id: h.id, text: h.textContent ?? "", depth: h.tagName === "H3" ? 3 : 2 })),
      );
    read();
    // the page is lazy: read again when it mounts.
    const watch = new MutationObserver(read);
    watch.observe(main, { childList: true, subtree: true });
    return () => watch.disconnect();
  }, [pathname]);

  if (!headings.length) return null;
  return (
    <aside className="sticky top-16 hidden max-h-[calc(100vh-4rem)] self-start overflow-y-auto pt-4 pb-10 [scrollbar-width:thin] xl:block">
      <div className="flex items-center gap-2 text-sm font-medium text-gray-600 dark:text-gray-300">
        <List size={16} className="shrink-0" />
        On this page
      </div>
      <nav className="mt-2">
        {headings.map((heading) => (
          <Item key={heading.id} heading={heading} />
        ))}
      </nav>
    </aside>
  );
}

/** One line of the table. Task 04 turns `active` on as the page scrolls. */
function Item({ heading, active = false }: { heading: Heading; active?: boolean }) {
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
