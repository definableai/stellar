import meta from "./meta.ts";
import mdx from "@mdx-js/rollup";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import rehypeAutolinkHeadings from "rehype-autolink-headings";
import rehypeSlug from "rehype-slug";
import rehypePrettyCode from "rehype-pretty-code";
import { transformerNotationDiff } from "@shikijs/transformers";
import { codeMeta } from "./src/mdx/code-meta.ts";
import { mermaid } from "./src/mdx/mermaid.ts";
import remarkFrontmatter from "remark-frontmatter";
import remarkGfm from "remark-gfm";
import remarkMdxFrontmatter from "remark-mdx-frontmatter";
import { defineConfig, type PluginOption } from "vite";

const pages = mdx({
  providerImportSource: "@mdx-js/react",
  remarkPlugins: [remarkGfm, remarkFrontmatter, remarkMdxFrontmatter, mermaid],
  rehypePlugins: [
    rehypeSlug,
    // The anchor carries no text of its own: prose.css draws the "#",
    // so a heading's textContent stays the heading (Toc reads it).
    [rehypeAutolinkHeadings, {
      behavior: "append",
      content: [],
      properties: { className: ["anchor"], ariaHidden: true, tabIndex: -1 },
    }],
    // ==== task 02: rehype-pretty-code goes here ====
    // Both themes ride along, as CSS variables on every token; prose.css picks
    // one. Fences only: an inline `code` stays the prose's own.
    [rehypePrettyCode, {
      theme: { light: "github-light-default", dark: "dark-plus" },
      keepBackground: false,
      defaultLang: { block: "text" },
      transformers: [transformerNotationDiff()],
    }],
    codeMeta,
  ],
});

// @mdx-js/rollup drops the query before it filters, so it would compile
// `a.mdx?raw` as well and the source would stop being a string. Search reads it.
const mdxPlugin: PluginOption = {
  ...pages,
  enforce: "pre",
  transform(code: string, id: string) {
    return /[?&]raw(&|$)/.test(id) ? undefined : pages.transform.call(this, code, id);
  },
};

export default defineConfig({
  base: "/",
  plugins: [meta(), mdxPlugin, react(), tailwindcss()],
});
