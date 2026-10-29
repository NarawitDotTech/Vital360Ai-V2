import { ChevronRight, HeartPulse, History, LineChart, Lock, Trash2 } from "lucide-react";
import Link from "next/link";
import { PageTitle } from "@/components/PageTitle";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { getDict } from "@/i18n";

export default async function HomePage({ params }: { params: Promise<{ locale: string }> }) {
  const { locale } = await params;
  const t = getDict(locale);
  const more = [
    { href: `/${locale}/history/`, label: t.home.history, Icon: History },
    { href: `/${locale}/dashboard/`, label: t.home.dashboard, Icon: LineChart },
    { href: `/${locale}/delete/`, label: t.home.deleteData, Icon: Trash2 },
  ];
  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-col gap-2">
        <PageTitle>{t.home.title}</PageTitle>
        <p className="text-muted-foreground">{t.home.intro}</p>
      </div>
      <Card>
        <CardContent className="gap-3">
          <Button asChild size="lg" className="w-full">
            <Link href={`/${locale}/capture/`}>
              <HeartPulse aria-hidden="true" />
              {t.home.start}
            </Link>
          </Button>
          <p className="text-center text-sm text-muted-foreground">{t.home.startHint}</p>
        </CardContent>
      </Card>
      <Alert role="note">
        <Lock aria-hidden="true" />
        <AlertDescription>{t.home.privacy}</AlertDescription>
      </Alert>
      <nav aria-label={t.home.moreTitle}>
        <Card className="gap-0 py-1">
          <ul>
            {more.map(({ href, label, Icon }, i) => (
              <li key={href}>
                {i > 0 && <Separator />}
                <Button asChild variant="ghost" className="h-14 w-full justify-start rounded-none px-5">
                  <Link href={href}>
                    <Icon aria-hidden="true" className="text-primary" />
                    <span className="flex-1 text-left">{label}</span>
                    <ChevronRight aria-hidden="true" className="text-muted-foreground" />
                  </Link>
                </Button>
              </li>
            ))}
          </ul>
        </Card>
      </nav>
    </div>
  );
}
