import { MDXProvider } from "@mdx-js/react";
import { useEffect, type ComponentType } from "react";
import { components } from "../mdx/components";
import { find, name } from "../site";

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
        {/* task 04: "Copy page" lands here */}
        <div id="page-actions" />
      </div>

      {front.description && <p className="mt-3 text-[18px] leading-7 text-gray-500 dark:text-gray-400">{front.description}</p>}

      <article className="prose mt-8">
        <MDXProvider components={components}>
          <Body />
        </MDXProvider>
      </article>

      {/* task 04: previous / next lands here */}
      <div id="page-footer" />
    </div>
  );
}
