import { MDXProvider } from "@mdx-js/react";
import { Check, Copy } from "lucide-react";
import { useEffect, useState, type ComponentType } from "react";
import { components } from "../mdx/components";
import { raw } from "../pages";
import { find, name } from "../site";
import Pagination from "./Pagination";

export type PageModule = { default: ComponentType; frontmatter?: Record<string, string> };

export default function Page({ mod, path }: { mod: PageModule; path: string }) {
  const front = mod.frontmatter ?? {};
  const here = find(path);
  const Body = mod.default;

  useEffect(() => {
    document.title = front.title ? `${front.title} - ${name}` : name;
    document.querySelector('meta[name="description"]')?.setAttribute("content", front.description ?? "");
  }, [front.title, front.description]);

  return (
    <div className="mx-auto max-w-[40.5rem] px-5 pt-8 pb-16">
      <div className="flex items-start justify-between gap-4">
        <div>
          {(here?.group ?? here?.tab) && (
            <div className="mb-1 text-sm font-medium text-primary dark:text-primary-light">{here?.group ?? here?.tab}</div>
          )}
          <h1 className="text-[36px] leading-9 font-medium tracking-[-0.5px] text-gray-900 dark:text-gray-200">{front.title}</h1>
        </div>
        <div id="page-actions">
          <CopyPage path={path} front={front} />
        </div>
      </div>

      {front.description && <p className="mt-3 text-[18px] leading-7 text-gray-500 dark:text-gray-400">{front.description}</p>}

      <article className="prose mt-8">
        <MDXProvider components={components}>
          <Body />
        </MDXProvider>
      </article>

      <div id="page-footer">
        <Pagination path={path} />
      </div>
    </div>
  );
}

/** The page as its writer typed it, with the frontmatter turned back into a title. */
function CopyPage({ path, front }: { path: string; front: Record<string, string> }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    const source = (await raw(path)?.()) ?? "";
    const header = `# ${front.title}\n\n${front.description}\n\n`;
    await navigator.clipboard.writeText(source.replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n+/, () => header));
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  return (
    <button
      type="button"
      onClick={copy}
      className="hidden items-center gap-2 rounded-xl border border-gray-200 px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-100 sm:flex dark:border-white/[0.07] dark:text-gray-300 dark:hover:bg-white/5"
    >
      {copied ? <Check size={16} /> : <Copy size={16} />}
      {copied ? "Copied" : "Copy page"}
    </button>
  );
}
