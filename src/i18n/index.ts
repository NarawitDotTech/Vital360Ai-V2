import { en } from "./en";
import { th, type Dict } from "./th";

export type { Dict };
export const locales = ["th", "en"] as const;
export type Locale = (typeof locales)[number];
export const defaultLocale: Locale = "th";

export const dictionaries: Record<Locale, Dict> = { th, en };

export function isLocale(v: string | undefined | null): v is Locale {
  return v === "th" || v === "en";
}

export function getDict(locale: string): Dict {
  return dictionaries[isLocale(locale) ? locale : defaultLocale];
}

/** Replace {name} placeholders. Unknown placeholders are left as-is. */
export function format(template: string, vars: Record<string, string | number>): string {
  return template.replace(/\{(\w+)\}/g, (m, k: string) => (k in vars ? String(vars[k]) : m));
}

/** BCP-47 tags for Intl / speech. */
export const bcp47: Record<Locale, string> = { th: "th-TH", en: "en-US" };
