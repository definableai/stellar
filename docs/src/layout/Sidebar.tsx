import { ChevronRight } from "lucide-react";
import { useEffect, useState } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { meta } from "../pages";
import { find, flatten, idOf, tabs, type Group } from "../site";

const row = "block w-full rounded-md py-1.5 pr-3 pl-4 text-left text-sm leading-5";
const idle = "text-gray-500 hover:bg-gray-950/[0.04] hover:text-gray-900 dark:text-gray-400 dark:hover:bg-white/[0.04] dark:hover:text-gray-200";
const on = "font-medium text-primary dark:text-primary-light";

/** The page's own label, or a readable one until the page is written. */
function label(page: string) {
  const front = meta(page);
  const stem = page.split("/").pop()!.replace(/-/g, " ");
  return front.sidebarTitle || front.title || stem[0].toUpperCase() + stem.slice(1);
}

/** The tab's groups. In the column at `lg`, inside the drawer below it. */
export default function Sidebar({ drawer = false }: { drawer?: boolean }) {
  const { pathname } = useLocation();
  const here = idOf(pathname);
  const tab = tabs.find((t) => t.tab === find(pathname)?.tab) ?? tabs[0];
  return (
    <aside
      className={
        drawer
          ? "px-5 pt-5 pb-10"
          : "sticky top-12 hidden h-[calc(100vh-3rem)] overflow-y-auto px-5 pt-5 pb-10 [scrollbar-width:thin] lg:block"
      }
    >
      {tab.groups.map((group) => (
        <div key={group.group} className="mt-8 first:mt-0">
          <div className="mb-2 pl-4 text-sm font-semibold text-gray-900 dark:text-gray-200">{group.group}</div>
          <List pages={group.pages} here={here} />
        </div>
      ))}
    </aside>
  );
}

function List({ pages, here, nested = false }: { pages: (string | Group)[]; here: string; nested?: boolean }) {
  return (
    <ul className={nested ? "pl-3" : ""}>
      {pages.map((page) => (
        <li key={typeof page === "string" ? page : page.group}>
          {typeof page === "string" ? (
            <NavLink to={`/${page}`} className={`${row} ${page === here ? on : idle}`}>
              {label(page)}
            </NavLink>
          ) : (
            <Nested group={page} here={here} />
          )}
        </li>
      ))}
    </ul>
  );
}

function Nested({ group, here }: { group: Group; here: string }) {
  const holds = flatten(group).includes(here);
  const [open, setOpen] = useState(holds);
  useEffect(() => { if (holds) setOpen(true); }, [holds]);
  return (
    <>
      <button type="button" onClick={() => setOpen(!open)} className={`${row} flex items-center ${idle}`}>
        {group.group}
        <ChevronRight size={14} className={`ml-1.5 transition-transform ${open ? "rotate-90" : ""}`} />
      </button>
      {open && <List pages={group.pages} here={here} nested />}
    </>
  );
}
