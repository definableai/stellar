/** Stubs. Task 03 makes these open, close and switch. */
import type { ReactNode } from "react";

type Props = { children?: ReactNode } & Record<string, unknown>;
const stub = (name: string) => (props: Props) => <div data-component={name}>{props.children}</div>;

export const Accordion = stub("Accordion");
export const AccordionGroup = stub("AccordionGroup");
export const Tabs = stub("Tabs");
export const Tab = stub("Tab");
export const Steps = stub("Steps");
export const Step = stub("Step");
export const Expandable = stub("Expandable");
