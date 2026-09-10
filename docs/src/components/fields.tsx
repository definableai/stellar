/** Stubs. Task 03 makes these the parameter and response rows. */
import type { ReactNode } from "react";

type Props = { children?: ReactNode } & Record<string, unknown>;
const stub = (name: string) => (props: Props) => <div data-component={name}>{props.children}</div>;

export const ParamField = stub("ParamField");
export const ResponseField = stub("ResponseField");
