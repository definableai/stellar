/** A ```mermaid fence, drawn. The library is large, so it arrives only on a page that has one. */
import { useEffect, useId, useState } from "react";

/** Our tokens under mermaid's names; it derives the rest (node fill, actor, activation) from these. */
const DARK = {
  darkMode: true,
  background: "transparent",
  primaryColor: "#171A18",        // gray-900, the node fill
  primaryTextColor: "#DFE1E0",    // gray-200
  primaryBorderColor: "#3F4140",  // gray-700
  nodeBorder: "#3F4140",          // the same, and it turns off v12's gradient stroke
  secondaryColor: "#262827",      // gray-800
  tertiaryColor: "#0B0C0E",       // codeblock, the box behind a subgraph
  lineColor: "#9FA1A0",           // gray-400
  arrowheadColor: "#9FA1A0",
  textColor: "#DFE1E0",
  edgeLabelBackground: "#111414", // the box's own colour, so a label hides the line under it and nothing else
  noteBkgColor: "#262827",
  noteBorderColor: "#3F4140",
  noteTextColor: "#DFE1E0",
};

const LIGHT = {
  darkMode: false,
  background: "transparent",
  primaryColor: "#FFFFFF",        // the box is already gray-50, so a node lifts off it in white
  primaryTextColor: "#171A18",    // gray-900
  primaryBorderColor: "#CED1CF",  // gray-300
  nodeBorder: "#CED1CF",
  secondaryColor: "#EEF1EF",      // gray-100
  tertiaryColor: "#EEF1EF",
  lineColor: "#707371",           // gray-500
  arrowheadColor: "#707371",
  textColor: "#171A18",
  edgeLabelBackground: "#F3F6F4",
  noteBkgColor: "#EEF1EF",
  noteBorderColor: "#CED1CF",
  noteTextColor: "#171A18",
};

const BOX =
  "my-6 overflow-x-auto rounded-2xl border border-gray-950/10 bg-gray-50 p-6 dark:border-white/10 dark:bg-white/[0.03]";
const DRAWN = "[&_svg]:h-auto [&_svg]:max-w-full";

export function Mermaid({ chart }: { chart: string }) {
  const id = "mermaid" + useId().replace(/\W/g, "");   // mermaid puts the id in a CSS selector
  const [svg, setSvg] = useState("");
  const [size, setSize] = useState({ natural: 0, floor: 0 });   // its own width, and the least it may shrink to
  const [error, setError] = useState("");

  useEffect(() => {
    let live = true;
    const draw = async () => {
      const { default: mermaid } = await import("mermaid");
      const dark = document.documentElement.classList.contains("dark");
      mermaid.initialize({
        startOnLoad: false,
        suppressErrorRendering: true,   // a bad fence is ours to show, not a bomb glued to <body>
        theme: "base",
        look: "classic",   // v12 defaults to "neo", which bevels every shape; the site is flat
        fontFamily: "Inter Variable, Inter, sans-serif",
        themeVariables: dark ? DARK : LIGHT,
      });
      try {
        const drawn = await mermaid.render(id, chart);
        if (!live) return;
        // Mermaid scales the picture down to fit; below 72% of its natural width the labels
        // stop being readable, so from there the box scrolls instead.
        const natural = Number(/viewBox="[\d.]+ [\d.]+ ([\d.]+)/.exec(drawn.svg)?.[1] ?? 0);
        setSize({ natural: Math.ceil(natural), floor: Math.round(natural * 0.72) });
        setSvg(drawn.svg);
        setError("");
      } catch (e) {
        if (!live) return;
        setSvg("");
        setError(e instanceof Error ? e.message : String(e));
      }
    };
    void draw();
    const watch = new MutationObserver(() => void draw());   // the theme toggle writes `class` on <html>
    watch.observe(document.documentElement, { attributeFilter: ["class"] });
    return () => {
      live = false;
      watch.disconnect();
    };
  }, [chart, id]);

  // A broken diagram is seen, not swallowed: the message, then the fence as written.
  if (error) {
    return (
      <div className={BOX}>
        <p className="my-0 text-sm text-red-600 dark:text-red-400">{error.split("\n")[0]}</p>
        <pre className="mt-3 mb-0 font-mono text-sm leading-6 text-gray-600 dark:text-gray-400">{chart}</pre>
      </div>
    );
  }
  return (
    <div className={`${BOX} ${DRAWN} ${svg ? "" : "min-h-32"}`}>
      {/* As wide as the drawing, centred; never narrower than the floor, so a wide one scrolls right and its left edge stays reachable. */}
      <div className="mx-auto max-w-full" style={{ width: size.natural || undefined, minWidth: size.floor }} dangerouslySetInnerHTML={{ __html: svg }} />
    </div>
  );
}
