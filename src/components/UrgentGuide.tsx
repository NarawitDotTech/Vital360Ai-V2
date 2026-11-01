"use client";

import { TriangleAlert } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { useI18n } from "./I18nProvider";

export function UrgentGuide() {
  const { t } = useI18n();
  return (
    <Alert
      variant="destructive"
      role="note"
      aria-labelledby="urgent-title"
      data-testid="urgent-guide"
    >
      <TriangleAlert aria-hidden="true" />
      <AlertTitle>
        <h2 id="urgent-title" className="font-semibold">
          {t.urgent.title}
        </h2>
      </AlertTitle>
      <AlertDescription>
        <p>{t.urgent.intro}</p>
        <ul className="mt-2 list-disc space-y-1 pl-5">
          {t.urgent.items.map((it) => (
            <li key={it}>{it}</li>
          ))}
        </ul>
      </AlertDescription>
    </Alert>
  );
}
