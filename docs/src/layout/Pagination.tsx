import { Link } from "react-router-dom";
import { meta } from "../pages";
import { find } from "../site";

const link = "text-sm text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white";

/** The page before and the page after, in the tab's reading order. */
export default function Pagination({ path }: { path: string }) {
  const here = find(path);
  if (!here?.prev && !here?.next) return null;
  const title = (page: string) => meta(page).title || page;
  return (
    <nav className="mt-12 flex items-center justify-between gap-4 border-t border-gray-950/10 pt-6 dark:border-white/10">
      {here?.prev ? <Link to={`/${here.prev}`} className={link}>← {title(here.prev)}</Link> : <span />}
      {here?.next && <Link to={`/${here.next}`} className={link}>{title(here.next)} →</Link>}
    </nav>
  );
}
