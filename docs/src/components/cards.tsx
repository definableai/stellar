/** Stubs. Task 03 makes these the cards and their grids. */
import type { ReactNode } from "react";

type Props = { children?: ReactNode } & Record<string, unknown>;
const stub = (name: string) => (props: Props) => <div data-component={name}>{props.children}</div>;

export const Card = stub("Card");
export const CardGroup = stub("CardGroup");
export const Columns = stub("Columns");
