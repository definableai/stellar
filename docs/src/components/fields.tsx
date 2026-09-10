/** One row of a reference table: a name, what it is, and what it means. */
import type { ReactNode } from "react";
import { inner } from "./misc";

type FieldProps = { type?: string; required?: boolean; default?: string; deprecated?: boolean; children?: ReactNode };

function Field({ name, type, required, default: fallback, deprecated, children }: FieldProps & { name?: string }) {
  return (
    <div className="border-b border-gray-100 py-4 last:border-0 dark:border-white/[0.06]">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className={`font-mono text-sm font-semibold text-gray-900 dark:text-gray-200${deprecated ? " line-through" : ""}`}>{name}</span>
        {type && <span className="font-mono text-xs text-gray-500">{type}</span>}
        {required && <span className="text-xs text-red-500">required</span>}
        {fallback !== undefined && <span className="text-xs text-gray-500">default: {fallback}</span>}
      </div>
      {children != null && <div className={`mt-2 text-sm leading-6 text-gray-600 dark:text-gray-400 ${inner}`}>{children}</div>}
    </div>
  );
}

/** Mintlify names a parameter by where it rides; the word itself is not shown. */
export function ParamField({ path, query, body, header, ...rest }: FieldProps & { path?: string; query?: string; body?: string; header?: string }) {
  return <Field name={path ?? query ?? body ?? header} {...rest} />;
}

export function ResponseField({ name, ...rest }: FieldProps & { name?: string }) {
  return <Field name={name} {...rest} />;
}
