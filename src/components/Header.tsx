import Link from "next/link";
import { BrandLogo } from "./BrandLogo";
import { LanguageSwitch } from "./LanguageSwitch";
import type { Dict, Locale } from "@/i18n";

export function Header({ locale, t }: { locale: Locale; t: Dict }) {
  return (
    <header className="border-b bg-card">
      <div className="mx-auto flex max-w-2xl items-center justify-between gap-3 px-4 py-2">
        <Link
          href={`/${locale}/`}
          aria-label={t.common.brandHome}
          className="flex min-h-12 items-center gap-2.5 rounded-lg text-foreground no-underline"
        >
          <BrandLogo />
          <span className="text-lg leading-tight font-semibold">{t.meta.appName}</span>
        </Link>
        <LanguageSwitch />
      </div>
    </header>
  );
}
