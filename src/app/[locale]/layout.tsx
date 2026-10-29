import type { Metadata, Viewport } from "next";
import { notFound } from "next/navigation";
import type { ReactNode } from "react";
import "../globals.css";
import { Header } from "@/components/Header";
import { I18nProvider } from "@/components/I18nProvider";
import { ServiceWorkerRegister } from "@/components/ServiceWorkerRegister";
import { getDict, isLocale, locales } from "@/i18n";

type Props = { children: ReactNode; params: Promise<{ locale: string }> };

export function generateStaticParams() {
  return locales.map((locale) => ({ locale }));
}

const SITE = process.env.NEXT_PUBLIC_SITE_URL ?? "https://vitals360.pages.dev";

export async function generateMetadata({ params }: Pick<Props, "params">): Promise<Metadata> {
  const { locale } = await params;
  const t = getDict(locale);
  return {
    metadataBase: new URL(SITE),
    title: { default: `${t.meta.appName} | ${t.meta.tagline}`, template: `%s | ${t.meta.appName}` },
    description: t.meta.description,
    manifest: "/manifest.webmanifest",
    icons: {
      icon: [{ url: "/favicon-32.png", sizes: "32x32", type: "image/png" }, { url: "/icon.svg", type: "image/svg+xml" }],
      apple: [{ url: "/apple-touch-icon.png", sizes: "180x180" }],
    },
    appleWebApp: { capable: true, title: t.meta.appName, statusBarStyle: "default" },
    openGraph: {
      title: t.meta.appName,
      description: t.meta.description,
      type: "website",
      locale: locale === "th" ? "th_TH" : "en_US",
      images: [{ url: "/og.png", width: 1200, height: 630, alt: t.meta.appName }],
    },
    twitter: { card: "summary_large_image", title: t.meta.appName, description: t.meta.description, images: ["/og.png"] },
  };
}

export const viewport: Viewport = { themeColor: "#0b6e72", width: "device-width", initialScale: 1 };

export default async function LocaleLayout({ children, params }: Props) {
  const { locale } = await params;
  if (!isLocale(locale)) notFound();
  const t = getDict(locale);
  return (
    <html lang={locale}>
      <body>
        <a
          className="sr-only rounded-lg bg-primary px-4 py-3 text-primary-foreground focus:not-sr-only focus:fixed focus:top-4 focus:left-4 focus:z-50"
          href="#main"
        >
          {t.common.skipToContent}
        </a>
        <I18nProvider locale={locale} dict={t}>
          <Header locale={locale} t={t} />
          <main id="main" className="mx-auto w-full max-w-2xl flex-1 px-4 pt-6 pb-10">
            {children}
          </main>
          <footer className="mx-auto w-full max-w-2xl px-4 pb-8 text-sm text-muted-foreground">{t.common.footerNote}</footer>
          <ServiceWorkerRegister />
        </I18nProvider>
      </body>
    </html>
  );
}
