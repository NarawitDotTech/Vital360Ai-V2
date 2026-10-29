"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useId, useState } from "react";
import { format } from "@/i18n";
import { CONSENT_VERSION, makeConsentRecord } from "@/storage/consent-gate";
import { saveConsent } from "@/storage/db";
import { useI18n } from "./I18nProvider";

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
    <div className="stack">
      <h1>{t.consent.title}</h1>
      <p>{t.consent.intro}</p>
      <section className="card card-tinted" aria-labelledby="consent-what">
        <h2 id="consent-what">{t.consent.whatTitle}</h2>
        <ul className="mt-2 list-disc space-y-2 pl-6">
          <li>{t.consent.point1}</li>
          <li>
            <strong>{t.consent.point2}</strong>
          </li>
          <li>{t.consent.point3}</li>
          <li>{t.consent.point4}</li>
        </ul>
      </section>
      <section className="notice notice-warn" aria-labelledby="consent-important">
        <h2 id="consent-important" className="!text-lg">
          {t.consent.notMedicalTitle}
        </h2>
        <p>{t.consent.notMedical}</p>
      </section>
      <div className="field-check">
        <input
          id={checkId}
          type="checkbox"
          checked={checked}
          onChange={(e) => {
            setChecked(e.target.checked);
            if (e.target.checked) setError(null);
          }}
          aria-describedby={error ? `${checkId}-err` : undefined}
        />
        <label htmlFor={checkId}>{t.consent.checkbox}</label>
      </div>
      {error && (
        <p id={`${checkId}-err`} role="alert" className="notice notice-danger">
          {error}
        </p>
      )}
      <button type="button" className="btn btn-primary btn-big" onClick={agree} disabled={busy}>
        {t.consent.agree}
      </button>
      <Link href={`/${locale}/`} className="btn btn-secondary">
        {t.consent.decline}
      </Link>
      <p className="text-sm text-muted">{format(t.consent.version, { version: CONSENT_VERSION })}</p>
    </div>
  );
}
