import Link from "next/link";
import { getDict } from "@/i18n";

export default async function HomePage({ params }: { params: Promise<{ locale: string }> }) {
  const { locale } = await params;
  const t = getDict(locale);
  return (
    <div className="stack">
      <h1>{t.home.title}</h1>
      <p>{t.home.intro}</p>
      <div className="stack-sm">
        <Link href={`/${locale}/capture/`} className="btn btn-primary btn-big">
          <span aria-hidden="true">▶</span>
          {t.home.start}
        </Link>
        <p className="text-center text-muted">{t.home.startHint}</p>
      </div>
      <p className="notice">{t.home.privacy}</p>
      <nav aria-label={t.home.moreTitle} className="stack-sm">
        <Link href={`/${locale}/history/`} className="btn btn-secondary">
          {t.home.history}
        </Link>
        <Link href={`/${locale}/dashboard/`} className="btn btn-secondary">
          {t.home.dashboard}
        </Link>
        <Link href={`/${locale}/delete/`} className="btn btn-secondary">
          {t.home.deleteData}
        </Link>
      </nav>
    </div>
  );
}
