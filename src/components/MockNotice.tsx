"use client";

import { FlaskConical } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { useI18n } from "./I18nProvider";

export function isMockModel(modelVersion: string): boolean {
  return modelVersion.startsWith("mock");
}

/** Compact amber notice for results that come from the placeholder (mock) model. */
export function MockNotice() {
  const { t } = useI18n();
  return (
    <Alert variant="warning" role="note" data-testid="mock-notice">
      <FlaskConical aria-hidden="true" />
      <AlertTitle className="font-semibold">{t.results.mockTitle}</AlertTitle>
      <AlertDescription>{t.results.mockText}</AlertDescription>
    </Alert>
  );
}
