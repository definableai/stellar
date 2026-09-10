import { Search as SearchIcon } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { index, search, type Section } from "../search";

const kbd = "rounded border border-gray-300 px-1.5 py-0.5 font-mono text-[11px] text-gray-500 dark:border-gray-700 dark:text-gray-500";

/** The ⌘K modal. Mounted only while open: it focuses itself and loads the index. */
export default function Search({ close }: { close: () => void }) {
  const [query, setQuery] = useState("");
  const [sections, setSections] = useState<Section[]>([]);
  const [at, setAt] = useState(0);
  const list = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  useEffect(() => { index().then(setSections); }, []);

  const hits = useMemo(() => search(query, sections), [query, sections]);
  useEffect(() => setAt(0), [query]);
  useEffect(() => { list.current?.children[at]?.scrollIntoView({ block: "nearest" }); }, [at, hits]);

  const open = (hit: Section) => { navigate(`/${hit.page.path}${hit.anchor ? `#${hit.anchor}` : ""}`); close(); };

  // No dependency list: the listener is rebound each render, so it reads today's hits.
  useEffect(() => {
    const press = (event: KeyboardEvent) => {
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        setAt((now) => (hits.length ? (now + (event.key === "ArrowDown" ? 1 : hits.length - 1)) % hits.length : 0));
      }
      if (event.key === "Enter" && hits[at]) open(hits[at]);
    };
    addEventListener("keydown", press);
    return () => removeEventListener("keydown", press);
  });

  return (
    <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm" onMouseDown={close}>
      <div
        className="mx-auto mt-[12vh] w-[min(640px,92vw)] overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-2xl dark:border-white/10 dark:bg-background-dark"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="flex h-14 items-center gap-3 px-4">
          <SearchIcon size={18} className="shrink-0 text-gray-400 dark:text-gray-500" />
          <input
            autoFocus
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search…"
            className="min-w-0 flex-1 bg-transparent text-base text-gray-900 outline-none placeholder:text-gray-400 dark:text-gray-100 dark:placeholder:text-gray-500"
          />
          <kbd className={kbd}>esc</kbd>
        </div>

        {query && hits.length > 0 && (
          <div ref={list} className="max-h-[60vh] overflow-auto border-t border-gray-200 py-2 dark:border-white/10">
            {hits.map((hit, i) => (
              <button
                key={`${hit.page.path}#${hit.anchor}`}
                type="button"
                onMouseMove={() => setAt(i)}
                onClick={() => open(hit)}
                className={`flex w-full flex-col px-4 py-2.5 text-left ${i === at ? "bg-gray-950/[0.04] dark:bg-white/[0.05]" : ""}`}
              >
                <span className={`text-sm ${i === at ? "text-primary dark:text-primary-light" : "text-gray-900 dark:text-gray-200"}`}>
                  {hit.heading || hit.page.title}
                </span>
                <span className="text-xs text-gray-500">{hit.page.group} › {hit.page.title}</span>
              </button>
            ))}
          </div>
        )}

        {query && hits.length === 0 && sections.length > 0 && (
          <div className="border-t border-gray-200 px-4 py-10 text-center text-sm text-gray-500 dark:border-white/10">No results for “{query}”</div>
        )}

        <div className="flex h-10 items-center gap-4 border-t border-gray-200 px-4 text-xs text-gray-500 dark:border-white/10">
          <span className="flex items-center gap-1.5"><kbd className={kbd}>↑↓</kbd> navigate</span>
          <span className="flex items-center gap-1.5"><kbd className={kbd}>↵</kbd> open</span>
          <span className="flex items-center gap-1.5"><kbd className={kbd}>esc</kbd> close</span>
        </div>
      </div>
    </div>
  );
}
