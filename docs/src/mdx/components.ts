/** What an MDX page may say: the tags that need React, and every component by name. */
import { createElement, type ComponentProps } from "react";
import { Link } from "react-router-dom";
import { Callout, Check, Danger, Info, Note, Tip, Warning } from "../components/callouts";
import { Card, CardGroup, Columns } from "../components/cards";
import { CodeBlock, CodeGroup } from "../components/code";
import { Accordion, AccordionGroup, Expandable, Step, Steps, Tab, Tabs } from "../components/disclosure";
import { ParamField, ResponseField } from "../components/fields";
import { Frame, Icon, Tooltip } from "../components/misc";

/** A link: the router for a site path, a new tab for the web, plain for an anchor. */
function A({ href = "", ...rest }: ComponentProps<"a">) {
  if (/^(https?:|mailto:)/.test(href)) return createElement("a", { href, target: "_blank", rel: "noreferrer", ...rest });
  if (href.startsWith("#")) return createElement("a", { href, ...rest });
  return createElement(Link, { to: href, ...rest });
}

const Table = (props: ComponentProps<"table">) =>
  createElement("div", { className: "my-5 overflow-x-auto" }, createElement("table", props));

export const components = {
  a: A,
  pre: CodeBlock,
  table: Table,
  Note, Tip, Info, Warning, Check, Danger, Callout,
  Card, CardGroup, Columns,
  Accordion, AccordionGroup, Tabs, Tab, Steps, Step, Expandable,
  ParamField, ResponseField,
  Frame, Tooltip, Icon,
  CodeGroup,
};
