/** The frame, the tooltip, and any lucide icon an `icon="…"` prop names. */
import { DynamicIcon, iconNames, type IconName } from "lucide-react/dynamic";
import type { ReactNode } from "react";

/**
 * The children of a component are MDX, so prose.css has already given them
 * their margins. prose.css is unlayered and Tailwind's utilities are in
 * `@layer utilities`, which loses whatever the specificity — `!` is how a
 * component takes its own spacing back.
 */
export const inner = "[&>:first-child]:mt-0! [&>:last-child]:mb-0!";

type IconProps = { icon?: string; size?: number | string; color?: string; className?: string };

const known = new Set<string>(iconNames);

/** A lucide icon by name. A name lucide does not have draws nothing — DynamicIcon would log. */
export function Icon({ icon, size = 16, color, className = "" }: IconProps) {
  if (!icon || !known.has(icon)) return null;
  return (
    <DynamicIcon
      name={icon as IconName}
      size={size}
      color={color}
      aria-hidden
      className={`inline-block shrink-0 align-[-0.15em] ${className}`}
    />
  );
}

export function Frame({ caption, children, className = "" }: { caption?: ReactNode; children?: ReactNode; className?: string }) {
  return (
    <div className={`my-6 rounded-2xl border border-gray-950/10 bg-gray-50 p-1 text-center dark:border-white/10 dark:bg-white/[0.03] ${className}`}>
      <div className="[&_img]:my-0! [&_img]:rounded-xl">{children}</div>
      {caption != null && <div className="mt-2 mb-1 text-xs text-gray-500">{caption}</div>}
    </div>
  );
}

/** Hover, no JavaScript: the popover is a sibling the group's hover reveals. */
export function Tooltip({ tip, children }: { tip?: ReactNode; children?: ReactNode }) {
  return (
    <span className="group relative inline-block">
      <span className="cursor-help underline decoration-dotted underline-offset-4">{children}</span>
      <span
        role="tooltip"
        className="pointer-events-none absolute bottom-full left-1/2 z-20 mb-2 hidden w-max max-w-xs -translate-x-1/2 rounded-lg bg-gray-900 px-3 py-2 text-xs leading-4 font-normal text-gray-100 shadow-lg ring-1 ring-black/10 group-hover:block dark:bg-gray-800 dark:text-gray-200 dark:ring-white/10"
      >
        {tip}
      </span>
    </span>
  );
}
