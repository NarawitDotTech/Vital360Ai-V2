"use client";

import { ChevronLeft } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Button } from "@/components/ui/button";
import { useI18n } from "./I18nProvider";

/** Logical parent of a page (not browser history, which may be empty after a refresh or deep link). */
export function parentPath(locale: string, pathname: string): string | null {
  const rest = pathname.replace(/^\/(th|en)(?=\/|$)/, "").replace(/\/+$/, "");
  if (!rest) return null; // home
  if (/^\/capture\/[^/]+$/.test(rest) || rest === "/results") return `/${locale}/capture/`;
  return `/${locale}/`;
}

export function BackButton() {
  const { locale, t } = useI18n();
  const href = parentPath(locale, usePathname() || "");
  if (!href) return null;
  return (
    <Button asChild variant="ghost" size="sm" className="-ml-2">
      <Link href={href} aria-label={t.common.back}>
        <ChevronLeft aria-hidden="true" className="size-5" />
        <span className="sr-only sm:not-sr-only">{t.common.back}</span>
      </Link>
    </Button>
  );
}
