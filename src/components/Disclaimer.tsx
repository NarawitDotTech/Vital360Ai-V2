"use client";

import { Info } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { useI18n } from "./I18nProvider";

export function Disclaimer() {
  const { t } = useI18n();
  return (
    <Alert
      role="note"
      aria-labelledby="disclaimer-title"
      data-testid="disclaimer"
    >
      <Info aria-hidden="true" />
      <AlertTitle>
        <h2 id="disclaimer-title" className="font-semibold">
          {t.disclaimer.title}
        </h2>
      </AlertTitle>
      <AlertDescription>{t.disclaimer.text}</AlertDescription>
    </Alert>
  );
}
