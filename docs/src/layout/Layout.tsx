import { Menu } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { useLocation } from "react-router-dom";
import { meta } from "../pages";
import { find } from "../site";
import Search from "./Search";
import Sidebar from "./Sidebar";
import Toc from "./Toc";
import Topbar, { Tabs } from "./Topbar";

export default function Layout({ children }: { children: ReactNode }) {
  const { pathname } = useLocation();
  const [searching, setSearching] = useState(false);
  const [drawer, setDrawer] = useState(false);
  const here = find(pathname);

  useEffect(() => setDrawer(false), [pathname]);

  useEffect(() => {
    const press = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") { event.preventDefault(); setSearching(true); }
      if (event.key === "Escape") { setSearching(false); setDrawer(false); }
    };
    addEventListener("keydown", press);
    return () => removeEventListener("keydown", press);
  }, []);

  useEffect(() => {
    if (!searching && !drawer) return;
    const was = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = was; };
  }, [searching, drawer]);

  return (
    <>
      <Topbar search={() => setSearching(true)} />

      <div className="sticky top-12 z-20 flex h-12 items-center gap-3 border-b border-gray-200 bg-background-light/80 px-5 text-sm backdrop-blur lg:hidden dark:border-gray-800 dark:bg-background-dark/80">
        <button type="button" aria-label="Menu" onClick={() => setDrawer(true)} className="text-gray-500 dark:text-gray-400">
          <Menu size={18} />
        </button>
        <span className="truncate text-gray-500 dark:text-gray-400">
          {here?.group && `${here.group} › `}
          <span className="text-gray-900 dark:text-gray-200">{meta(pathname).title}</span>
        </span>
      </div>

      <div className="mx-auto grid max-w-[90rem] lg:grid-cols-[224px_minmax(0,1fr)] xl:grid-cols-[224px_minmax(0,1fr)_236px]">
        <Sidebar />
        <main id="content">{children}</main>
        <Toc />
      </div>

      {drawer && (
        <div className="fixed inset-0 z-40 bg-black/60 lg:hidden" onMouseDown={() => setDrawer(false)}>
          <div
            className="fixed inset-y-0 left-0 w-72 overflow-y-auto border-r border-gray-200 bg-background-light [scrollbar-width:thin] dark:border-gray-800 dark:bg-background-dark"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <Tabs className="flex flex-col items-start gap-1 border-b border-gray-200 p-3 md:hidden dark:border-gray-800" />
            <Sidebar drawer />
          </div>
        </div>
      )}

      {searching && <Search close={() => setSearching(false)} />}
    </>
  );
}
