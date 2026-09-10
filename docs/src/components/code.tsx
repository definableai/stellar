/** Every fence: the frame, the header, the copy button, and the meta's flags. */
import { Check, ChevronDown, Copy } from "lucide-react";
import { Children, isValidElement, useRef, useState, type ComponentProps, type ReactNode, type RefObject } from "react";
import { parseMeta, type Meta } from "../mdx/code-meta";

/** The `pre` rehype-pretty-code hands over, with the fence's meta still on it. */
type PreProps = ComponentProps<"pre"> & { "data-meta"?: string };

const FRAME =
  "group relative mt-5 mb-8 overflow-hidden rounded-2xl border border-gray-950/10 bg-gray-50 dark:border-white/10 dark:bg-codeblock";
const HEADER = "flex h-10 items-center gap-4 border-b border-gray-200 px-4 dark:border-white/[0.08]";
const TITLE = "text-[13px] font-medium";

export function CodeBlock(props: PreProps) {
  const meta = parseMeta(props["data-meta"]);
  const frame = useRef<HTMLDivElement>(null);
  const copy = meta.nocopy ? null : <CopyButton frame={frame} />;
  return (
    <div ref={frame} className={FRAME}>
      {meta.title || meta.run ? (
        <div className={HEADER}>
          <span className={`truncate ${TITLE} text-gray-500 dark:text-gray-400`}>{meta.title}</span>
          <span className="ml-auto flex items-center gap-3">
            {meta.run && <Tested />}
            {copy}
          </span>
        </div>
      ) : (
        // No header: the button floats over the code, on hover.
        copy && (
          <div className="absolute top-2 right-2 z-10 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
            {copy}
          </div>
        )
      )}
      <Body meta={meta} {...props} />
    </div>
  );
}

/** Several fences, one frame: their titles are the tabs, one body is shown. */
export function CodeGroup({ children }: { children?: ReactNode }) {
  const blocks = Children.toArray(children).filter(isValidElement<PreProps>);
  const frame = useRef<HTMLDivElement>(null);
  const [tab, setTab] = useState(0);
  if (!blocks.length) return null;

  const at = Math.min(tab, blocks.length - 1);
  const metas = blocks.map((block) => parseMeta(block.props["data-meta"]));
  return (
    <div ref={frame} className={FRAME}>
      <div className={HEADER}>
        <div className="flex h-full items-center gap-4 overflow-x-auto [scrollbar-width:none]">
          {blocks.map((block, i) => (
            <button
              key={block.key ?? i}
              type="button"
              onClick={() => setTab(i)}
              className={`relative h-full whitespace-nowrap ${TITLE} ${
                i === at ? "text-gray-900 dark:text-white" : "text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
              }`}
            >
              {metas[i].title || `Tab ${i + 1}`}
              {i === at && <span className="absolute inset-x-0 -bottom-px h-0.5 bg-primary dark:bg-primary-light" />}
            </button>
          ))}
        </div>
        {!metas[at].nocopy && (
          <span className="ml-auto">
            <CopyButton frame={frame} />
          </span>
        )}
      </div>
      <Body key={at} meta={metas[at]} {...blocks[at].props} />
    </div>
  );
}

/** The code itself: shiki's spans, the flags as attributes, the expand bar. */
function Body({ meta, className, ...rest }: PreProps & { meta: Meta }) {
  const [open, setOpen] = useState(!meta.expandable);
  return (
    <div className="relative">
      {/* 16 lines of 24px, plus the pre's own top padding. */}
      <div className={open ? undefined : "max-h-100 overflow-hidden"}>
        <pre
          className={`shiki overflow-x-auto py-4 font-mono text-sm leading-6 ${className ?? ""}`}
          data-lines={meta.lines ? "" : undefined}
          data-wrap={meta.wrap ? "" : undefined}
          {...rest}
        />
      </div>
      {meta.expandable && (
        <>
          {!open && (
            <div className="pointer-events-none absolute inset-x-0 bottom-9 h-16 bg-linear-to-t from-gray-50 dark:from-codeblock" />
          )}
          <button
            type="button"
            onClick={() => setOpen(!open)}
            className="relative flex h-9 w-full items-center justify-center gap-1.5 border-t border-gray-200 text-[13px] font-medium text-gray-500 hover:text-gray-900 dark:border-white/[0.08] dark:text-gray-400 dark:hover:text-white"
          >
            <ChevronDown size={14} className={open ? "rotate-180" : undefined} />
            {open ? "Collapse" : "Expand"}
          </button>
        </>
      )}
    </div>
  );
}

/** check.py --run executed this fence and it printed what the page says. */
const Tested = () => (
  <span className="flex items-center gap-1.5 text-xs text-gray-500 dark:text-gray-400" title="check.py --run ran this example">
    <span className="h-1.5 w-1.5 rounded-full bg-green-500" />
    Tested
  </span>
);

/** The line numbers are drawn by CSS, so textContent is the code and nothing else. */
function CopyButton({ frame }: { frame: RefObject<HTMLDivElement | null> }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      aria-label="Copy"
      onClick={() => {
        void navigator.clipboard?.writeText(frame.current?.querySelector("pre")?.textContent ?? "");
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      }}
      className="flex h-7 items-center gap-1.5 rounded-md px-1.5 text-xs font-medium text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white"
    >
      {copied ? <Check size={16} /> : <Copy size={16} />}
      {copied && "Copied"}
    </button>
  );
}
