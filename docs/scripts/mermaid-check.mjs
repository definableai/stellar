/** Every ```mermaid fence under content/ parses: `npm run check:diagrams`.
 *
 * A writer cannot see a diagram render, so this is the fence's own test. One
 * line per failure, `file:line: message`, and exit 1 on any.
 */
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

// Mermaid runs every label through DOMPurify, which wants a DOM. Under node it
// loads unsupported — no addHook, no sanitize — and mermaid dies on the first
// flowchart. Here are the three methods it calls, on the same module instance
// mermaid imports, so no jsdom. Ceiling: it leans on dompurify resolving to one
// hoisted copy; if that ever stops being true this file throws, it does not lie.
const { default: DOMPurify } = await import("dompurify");
DOMPurify.addHook ??= () => {};
DOMPurify.removeAllHooks ??= () => {};
DOMPurify.sanitize ??= (text) => String(text);

const { default: mermaid } = await import("mermaid");

const CONTENT = join(import.meta.dirname, "..", "content");
const FENCE = /^[ \t]*```mermaid[^\n]*\n([\s\S]*?)^[ \t]*```/gm;

let found = 0;
let seen = 0;
for (const file of readdirSync(CONTENT, { recursive: true })) {
  if (!file.endsWith(".mdx")) continue;
  const text = readFileSync(join(CONTENT, file), "utf8");
  for (const fence of text.matchAll(FENCE)) {
    seen++;
    const line = text.slice(0, fence.index).split("\n").length;
    try {
      await mermaid.parse(fence[1]);
    } catch (e) {
      found++;
      console.log(`content/${file}:${line}: ${(e.message ?? String(e)).split("\n")[0]}`);
    }
  }
}
console.log(found ? `${found} problem(s)` : `ok: ${seen} diagrams parse`);
process.exit(found ? 1 : 0);
