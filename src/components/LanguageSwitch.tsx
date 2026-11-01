"use client";

import { Languages } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Button } from "@/components/ui/button";
import { useI18n } from "./I18nProvider";

export function LanguageSwitch() {
  const { locale, t } = useI18n();
  const pathname = usePathname() || `/${locale}/`;
  const other = locale === "th" ? "en" : "th";
  const rest = pathname.replace(/^\/(th|en)(?=\/|$)/, "");
  const href = `/${other}${rest || "/"}`;
  return (
    <Button asChild variant="outline" size="sm">
      <Link
        href={href}
        hrefLang={other}
        lang={other}
        aria-label={t.header.switchLabel}
      >
        <Languages aria-hidden="true" />
        {t.header.switchText}
      </Link>
    </Button>
  );
}
