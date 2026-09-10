import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";
import "./styles/globals.css";
import "./styles/prose.css";

import { createRoot } from "react-dom/client";
import App from "./app";
import { colors } from "./site";
import { apply, get } from "./theme";

apply(get());

// docs.json owns the accent: the tokens in globals.css are only its defaults.
const root = document.documentElement.style;
root.setProperty("--color-primary", colors.primary);
root.setProperty("--color-primary-light", colors.light);
root.setProperty("--color-primary-dark", colors.dark);

createRoot(document.getElementById("root")!).render(<App />);
