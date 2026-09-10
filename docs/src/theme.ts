/** dark or light: what was chosen here, else docs.json, else the OS. */
import { appearance } from "./site";

export type Theme = "dark" | "light";

export function get(): Theme {
  let saved: string | null = null;
  try { saved = localStorage.getItem("theme"); } catch { /* storage off: fall through */ }
  const wanted = saved ?? appearance.default;
  if (wanted === "dark" || wanted === "light") return wanted;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export const apply = (theme: Theme) => document.documentElement.classList.toggle("dark", theme === "dark");

export function toggle(): Theme {
  const theme = get() === "dark" ? "light" : "dark";
  try { localStorage.setItem("theme", theme); } catch { /* storage off: this tab only */ }
  apply(theme);
  return theme;
}
