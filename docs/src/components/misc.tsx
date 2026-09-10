/** Stubs. Task 03 makes these the frame, the tooltip and the lucide icon. */
import type { ReactNode } from "react";

type Props = { children?: ReactNode } & Record<string, unknown>;
const stub = (name: string) => (props: Props) => <div data-component={name}>{props.children}</div>;
// Tooltip and Icon sit inside a sentence: a block would break the line.
const inline = (name: string) => (props: Props) => <span data-component={name}>{props.children}</span>;

export const Frame = stub("Frame");
export const Tooltip = inline("Tooltip");
export const Icon = inline("Icon");
