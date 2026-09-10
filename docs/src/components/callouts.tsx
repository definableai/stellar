/** The six named callouts and the custom one: one box, six colour rows. */
import { CircleAlert, CircleCheck, Info as InfoGlyph, Lightbulb, OctagonAlert, TriangleAlert, type LucideProps } from "lucide-react";
import type { ComponentType, ReactNode } from "react";
import { Icon } from "./misc";

type Props = { children?: ReactNode };

const box = "my-4 flex gap-3 rounded-2xl border px-5 py-4 text-sm leading-5";
// The text inside is MDX: take the outer margins back from prose.css.
const body =
  "min-w-0 flex-1 [&_p]:my-0 [&_p+p]:mt-2 [&_a]:text-inherit [&_a]:decoration-current [&_strong]:text-inherit [&>:first-child]:mt-0 [&>:last-child]:mb-0";

const gray = "border-gray-200 bg-gray-50 text-gray-700 dark:border-gray-800 dark:bg-gray-800/40 dark:text-gray-300";

const callout = (tone: string, Glyph: ComponentType<LucideProps>) =>
  function Callout({ children }: Props) {
    return (
      <div className={`${box} ${tone}`}>
        <Glyph size={16} aria-hidden className="mt-0.5 shrink-0" />
        <div className={body}>{children}</div>
      </div>
    );
  };

export const Note = callout("border-blue-200 bg-blue-50 text-blue-900 dark:border-blue-900 dark:bg-blue-600/20 dark:text-blue-300", CircleAlert);
export const Warning = callout("border-yellow-200 bg-yellow-50 text-yellow-900 dark:border-yellow-900 dark:bg-yellow-600/20 dark:text-yellow-300", TriangleAlert);
export const Info = callout(gray, InfoGlyph);
export const Tip = callout("border-green-200 bg-green-50 text-green-900 dark:border-green-900 dark:bg-green-600/20 dark:text-green-300", Lightbulb);
export const Check = callout("border-emerald-200 bg-emerald-50 text-emerald-900 dark:border-emerald-900 dark:bg-emerald-600/20 dark:text-emerald-300", CircleCheck);
export const Danger = callout("border-red-200 bg-red-50 text-red-900 dark:border-red-900 dark:bg-red-600/20 dark:text-red-300", OctagonAlert);

/** The one you colour yourself: a lucide name and a hex, tinted off that hex. */
export function Callout({ icon, color, children }: Props & { icon?: string; color?: string }) {
  const tint = color
    ? {
        color,
        borderColor: `color-mix(in oklab, ${color} 45%, transparent)`,
        backgroundColor: `color-mix(in oklab, ${color} 12%, transparent)`,
      }
    : undefined;
  return (
    <div className={`${box} ${color ? "" : gray}`} style={tint}>
      <Icon icon={icon} size={16} className="mt-0.5" />
      <div className={body}>{children}</div>
    </div>
  );
}
