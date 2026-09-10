/** The card, and the grid that holds a few of them. */
import { ArrowRight } from "lucide-react";
import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { Icon, inner } from "./misc";

type CardProps = {
  title?: ReactNode;
  icon?: string;
  href?: string;
  horizontal?: boolean;
  arrow?: boolean;
  cta?: ReactNode;
  children?: ReactNode;
  className?: string;
};

// `no-underline` and `text-inherit`: a card with an href is an <a>, and .prose would paint it as a link.
const box =
  "group relative block rounded-2xl border border-gray-950/10 bg-white px-6 py-5 text-inherit no-underline transition-colors hover:border-primary/40 dark:border-white/10 dark:bg-background-dark dark:hover:border-primary-light/40";

export function Card({ title, icon, href, horizontal, arrow, cta, children, className = "" }: CardProps) {
  const glyph = <Icon icon={icon} size={24} className="text-primary dark:text-primary-light" />;
  const words = (
    <>
      <div className={`text-base leading-6 font-semibold text-gray-900 dark:text-white ${arrow ? "pr-6" : ""}`}>{title}</div>
      {children != null && <div className={`mt-1 leading-6 text-gray-600 dark:text-gray-400 ${inner}`}>{children}</div>}
      {cta != null && (
        <div className="mt-3 inline-flex items-center gap-1 text-sm font-medium text-primary dark:text-primary-light">
          {cta}
          <ArrowRight size={14} aria-hidden />
        </div>
      )}
    </>
  );

  const guts = horizontal ? (
    <div className="flex items-start gap-4">
      {glyph}
      <div className="min-w-0 flex-1">{words}</div>
    </div>
  ) : (
    <>
      {icon && <div className="mb-3">{glyph}</div>}
      {words}
    </>
  );

  const all = (
    <>
      {guts}
      {arrow && (
        <ArrowRight size={16} aria-hidden className="absolute top-5 right-5 text-gray-400 transition-transform group-hover:translate-x-0.5" />
      )}
    </>
  );

  const style = `${box} ${className}`;
  if (!href) return <div className={style}>{all}</div>;
  if (/^(https?:|mailto:)/.test(href))
    return (
      <a href={href} target="_blank" rel="noreferrer" className={style}>
        {all}
      </a>
    );
  return (
    <Link to={href} className={style}>
      {all}
    </Link>
  );
}

const wide: Record<string, string> = { "1": "sm:grid-cols-1", "2": "sm:grid-cols-2", "3": "sm:grid-cols-3", "4": "sm:grid-cols-4" };

export function CardGroup({ cols = 2, children, className = "" }: { cols?: number | string; children?: ReactNode; className?: string }) {
  return <div className={`my-5 grid gap-4 ${wide[String(cols)] ?? wide["2"]} ${className}`}>{children}</div>;
}

/** Mintlify's other name for the same grid, for anything that is not a card. */
export const Columns = CardGroup;
