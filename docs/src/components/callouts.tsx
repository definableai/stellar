/** Stubs. Task 03 makes these the coloured callouts. */
import type { ReactNode } from "react";

type Props = { children?: ReactNode } & Record<string, unknown>;
const stub = (name: string) => (props: Props) => <div data-component={name}>{props.children}</div>;

export const Note = stub("Note");
export const Tip = stub("Tip");
export const Info = stub("Info");
export const Warning = stub("Warning");
export const Check = stub("Check");
export const Danger = stub("Danger");
export const Callout = stub("Callout");
