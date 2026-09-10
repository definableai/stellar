/** Stub. Task 02 makes these shiki, with titles, copy, highlights and tabs. */
import type { ComponentProps, ReactNode } from "react";

export function CodeBlock(props: ComponentProps<"pre">) {
  return (
    <div className="mt-5 mb-8 overflow-hidden rounded-2xl border border-gray-950/10 bg-gray-50 dark:border-white/10 dark:bg-codeblock">
      <pre className="overflow-x-auto p-4 font-mono text-sm leading-6" {...props} />
    </div>
  );
}

export function CodeGroup({ children }: { children?: ReactNode }) {
  return <div data-component="CodeGroup">{children}</div>;
}
