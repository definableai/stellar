/** What opens, closes and switches: accordions, tabs, steps, expandables. */
import { ChevronRight } from "lucide-react";
import { Children, isValidElement, useState, type ReactElement, type ReactNode } from "react";
import { Icon, inner } from "./misc";

/** The element children, whitespace between them dropped. */
const kids = <P,>(children: ReactNode) => Children.toArray(children).filter(isValidElement) as ReactElement<P>[];

const chevron = (open: boolean) => `shrink-0 transition-transform ${open ? "rotate-90" : ""}`;

type AccordionProps = {
  title?: ReactNode; description?: ReactNode; defaultOpen?: boolean; icon?: string; flush?: boolean; children?: ReactNode;
};

export function Accordion({ title, description, defaultOpen, icon, flush, children }: AccordionProps) {
  const [open, setOpen] = useState(!!defaultOpen);
  return (
    <div className={flush ? "" : "my-4 rounded-xl border border-gray-200 dark:border-white/10"}>
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-2 px-5 py-3.5 text-left text-sm font-medium text-gray-900 dark:text-gray-200"
      >
        <ChevronRight size={16} aria-hidden className={`${chevron(open)} text-gray-500`} />
        <Icon icon={icon} size={16} className="text-gray-500" />
        <span>
          {title}
          {description != null && <span className="mt-0.5 block text-xs font-normal text-gray-500">{description}</span>}
        </span>
      </button>
      {open && <div className={`px-5 pb-4 ${inner}`}>{children}</div>}
    </div>
  );
}

/** One border around the lot: the accordions inside are drawn flush, so they give up their own. */
export function AccordionGroup({ children }: { children?: ReactNode }) {
  return (
    <div className="my-4 divide-y divide-gray-200 rounded-xl border border-gray-200 dark:divide-white/10 dark:border-white/10">
      {kids<AccordionProps>(children).map((one, i) => (
        <Accordion key={i} {...one.props} flush />
      ))}
    </div>
  );
}

type TabProps = { title?: ReactNode; icon?: string; children?: ReactNode };

/** Tabs reads its children's props and draws them; only the live panel's body renders. */
export function Tabs({ children }: { children?: ReactNode }) {
  const tabs = kids<TabProps>(children);
  const [at, setAt] = useState(0);
  const live = tabs[at] ?? tabs[0];
  return (
    <div className="my-5">
      <div role="tablist" className="flex gap-6 border-b border-gray-200 text-sm dark:border-white/10">
        {tabs.map((tab, i) => (
          <button
            key={i}
            type="button"
            role="tab"
            aria-selected={i === at}
            onClick={() => setAt(i)}
            className={`-mb-px flex items-center gap-1.5 border-b-2 py-2.5 font-medium ${
              i === at
                ? "border-primary text-primary dark:border-primary-light dark:text-primary-light"
                : "border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
            }`}
          >
            <Icon icon={tab.props.icon} size={14} />
            {tab.props.title}
          </button>
        ))}
      </div>
      <div role="tabpanel" className={`pt-4 ${inner}`}>
        {live?.props.children}
      </div>
    </div>
  );
}

export const Tab = ({ children }: TabProps) => <div className={inner}>{children}</div>;

type StepProps = { title?: ReactNode; icon?: string; stepNumber?: number; children?: ReactNode };

/** Steps counts its children, so the connector knows where the last one is. */
export function Steps({ children }: { children?: ReactNode }) {
  const steps = kids<StepProps>(children);
  return (
    <div className="my-5">
      {steps.map(({ props }, i) => {
        const last = i === steps.length - 1;
        return (
          <div key={i} className="grid grid-cols-[28px_1fr] gap-x-4">
            <div className="flex flex-col items-center">
              <div className="flex h-7 w-7 items-center justify-center rounded-full bg-gray-100 text-xs font-semibold text-gray-700 dark:bg-white/10 dark:text-gray-200">
                {props.icon ? <Icon icon={props.icon} size={14} /> : (props.stepNumber ?? i + 1)}
              </div>
              {!last && <div className="w-px flex-1 bg-gray-200 dark:bg-white/10" />}
            </div>
            <div className={last ? "" : "pb-6"}>
              <div className="text-base leading-7 font-medium text-gray-900 dark:text-white">{props.title}</div>
              <div className={`mt-2 ${inner}`}>{props.children}</div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export const Step = ({ children }: StepProps) => <div className={inner}>{children}</div>;

export function Expandable({ title, defaultOpen, children }: { title?: ReactNode; defaultOpen?: boolean; children?: ReactNode }) {
  const [open, setOpen] = useState(!!defaultOpen);
  return (
    <div className="my-4">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 text-sm text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
      >
        <ChevronRight size={14} aria-hidden className={chevron(open)} />
        {open ? "Hide" : "Show"} {title}
      </button>
      {open && <div className={`mt-3 border-l border-gray-200 pl-4 dark:border-white/10 ${inner}`}>{children}</div>}
    </div>
  );
}
