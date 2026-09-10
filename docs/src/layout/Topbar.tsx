import { Moon, Search, Sun } from "lucide-react";
import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { find, flatten, name, navbar, tabs } from "../site";
import { get, toggle } from "../theme";

const round = "flex h-9 w-9 items-center justify-center rounded-full border border-gray-200 bg-gray-950/[0.02] text-gray-500 hover:text-gray-900 dark:border-gray-800 dark:bg-white/[0.03] dark:text-gray-400 dark:hover:text-gray-200";

export default function Topbar({ search }: { search: () => void }) {
  const [theme, setTheme] = useState(get);
  return (
    <header className="sticky top-0 z-30 border-b border-gray-200 bg-background-light/80 backdrop-blur dark:border-gray-800 dark:bg-background-dark/80">
      <div className="relative mx-auto flex h-12 max-w-[90rem] items-center px-5">
        <Link to="/" className="flex items-center gap-2 pl-4">
          <img src="/logo.svg" alt="" width="24" height="24" />
          <span className="text-[18px] font-semibold tracking-tight text-gray-900 dark:text-white">{name}</span>
        </Link>

        <Tabs className="absolute left-1/2 hidden -translate-x-1/2 items-center gap-1 md:flex" />

        <div className="ml-auto flex items-center gap-2">
          <button id="search" type="button" aria-label="Search" className={round} onClick={search}>
            <Search size={16} />
          </button>
          <button type="button" aria-label="Toggle theme" className={round} onClick={() => setTheme(toggle())}>
            {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
          </button>
          <Link
            to={navbar.primary.href}
            className="flex h-[34px] items-center rounded-full bg-primary-light px-4 text-sm font-medium text-gray-950 hover:opacity-80"
          >
            {navbar.primary.label}
          </Link>
        </div>
      </div>
    </header>
  );
}

/** The tab pills: centred in the topbar at `md`, stacked in the drawer below it. */
export function Tabs({ className }: { className: string }) {
  const here = find(useLocation().pathname)?.tab;
  return (
    <nav className={className}>
      {tabs.map((tab) => (
        <Link
          key={tab.tab}
          to={`/${flatten(tab)[0]}`}
          className={`flex h-9 items-center rounded-full px-4 text-sm font-medium ${
            tab.tab === here
              ? "bg-gray-950/5 text-gray-900 dark:bg-white/[0.06] dark:text-gray-100"
              : "text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-200"
          }`}
        >
          {tab.tab}
        </Link>
      ))}
    </nav>
  );
}
