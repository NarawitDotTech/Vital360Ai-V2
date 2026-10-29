"use client";

import { useI18n } from "./I18nProvider";

export function Disclaimer() {
  const { t } = useI18n();
  return (
    <aside className="notice" aria-labelledby="disclaimer-title" data-testid="disclaimer">
      <h2 id="disclaimer-title" className="!text-lg">
        {t.disclaimer.title}
      </h2>
      <p>{t.disclaimer.text}</p>
    </aside>
  );
}
