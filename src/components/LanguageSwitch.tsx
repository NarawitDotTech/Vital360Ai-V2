"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useI18n } from "./I18nProvider";

export function LanguageSwitch() {
  const { locale, t } = useI18n();
  const pathname = usePathname() || `/${locale}/`;
  const other = locale === "th" ? "en" : "th";
  const rest = pathname.replace(/^\/(th|en)(?=\/|$)/, "");
  const href = `/${other}${rest || "/"}`;
  return (
    <Link
      href={href}
      hrefLang={other}
      lang={other}
      aria-label={t.header.switchLabel}
      className="btn btn-secondary btn-inline !min-h-12 !py-1 !px-4 !text-base"
    >
      {t.header.switchText}
    </Link>
  );
}
