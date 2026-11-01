"use client";

import { Check, TriangleAlert } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useId, useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { format } from "@/i18n";
import { CONSENT_VERSION, makeConsentRecord } from "@/storage/consent-gate";
import { saveConsent } from "@/storage/db";
import { useI18n } from "./I18nProvider";
import { PageTitle } from "./PageTitle";

export function ConsentForm() {
  const { locale, t } = useI18n();
  const router = useRouter();
  const checkId = useId();
  const [checked, setChecked] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function agree() {
    if (!checked) {
      setError(t.consent.needCheck);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await saveConsent(makeConsentRecord(locale));
      router.push(`/${locale}/capture/`);
    } catch {
      setError(t.consent.saveFailed);
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col gap-5">
      <PageTitle>{t.consent.title}</PageTitle>
      <p>{t.consent.intro}</p>
      <Card aria-labelledby="consent-what" role="region">
        <CardHeader>
          <CardTitle>
            <h2 id="consent-what">{t.consent.whatTitle}</h2>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="list-disc space-y-2 pl-5">
            <li>{t.consent.point1}</li>
            <li>
              <strong className="font-semibold">{t.consent.point2}</strong>
            </li>
            <li>{t.consent.point3}</li>
            <li>{t.consent.point4}</li>
          </ul>
        </CardContent>
      </Card>
      <Alert variant="warning" role="note" aria-labelledby="consent-important">
        <TriangleAlert aria-hidden="true" />
        <AlertTitle>
          <h2 id="consent-important" className="font-semibold">
            {t.consent.notMedicalTitle}
          </h2>
        </AlertTitle>
        <AlertDescription>{t.consent.notMedical}</AlertDescription>
      </Alert>
      <Card size="sm">
        <CardContent>
          <div className="flex items-start gap-3">
            <Checkbox
              id={checkId}
              checked={checked}
              onCheckedChange={(v) => {
                setChecked(v === true);
                if (v === true) setError(null);
              }}
              aria-describedby={error ? `${checkId}-err` : undefined}
              className="mt-0.5"
            />
            <Label
              htmlFor={checkId}
              className="min-h-8 cursor-pointer items-start font-normal"
            >
              {t.consent.checkbox}
            </Label>
          </div>
        </CardContent>
      </Card>
      {error && (
        <Alert variant="destructive" id={`${checkId}-err`} role="alert">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
      <Button
        type="button"
        size="lg"
        className="w-full"
        onClick={agree}
        disabled={busy}
      >
        <Check aria-hidden="true" />
        {t.consent.agree}
      </Button>
      <Button asChild variant="ghost" size="lg" className="self-center">
        <Link href={`/${locale}/`}>{t.consent.decline}</Link>
      </Button>
      <p className="text-sm text-muted-foreground">
        {format(t.consent.version, { version: CONSENT_VERSION })}
      </p>
    </div>
  );
}
