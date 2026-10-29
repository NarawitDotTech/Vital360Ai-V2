"use client";

import { useI18n } from "./I18nProvider";

export function UrgentGuide() {
  const { t } = useI18n();
  return (
    <aside className="notice notice-danger" aria-labelledby="urgent-title" data-testid="urgent-guide">
      <h2 id="urgent-title" className="!text-lg">
        <span aria-hidden="true">⚠ </span>
        {t.urgent.title}
      </h2>
      <p>{t.urgent.intro}</p>
      <ul className="mt-2 list-disc space-y-1 pl-6">
        {t.urgent.items.map((it) => (
          <li key={it}>{it}</li>
        ))}
      </ul>
    </aside>
  );
}
