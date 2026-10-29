"use client";

import { Loader2 } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { hasValidConsent } from "@/storage/db";
import { useI18n } from "./I18nProvider";

/** Renders children only when a current consent record exists; otherwise redirects to the consent page. */
export function ConsentGate({ children }: { children: ReactNode }) {
  const { locale, t } = useI18n();
  const router = useRouter();
  const [state, setState] = useState<"checking" | "ok" | "denied">("checking");

  useEffect(() => {
    let cancelled = false;
    hasValidConsent().then((ok) => {
      if (cancelled) return;
      setState(ok ? "ok" : "denied");
      if (!ok) router.replace(`/${locale}/consent/`);
    });
    return () => {
      cancelled = true;
    };
  }, [locale, router]);

  if (state === "ok") return <>{children}</>;
  return (
    <Alert role="status">
      <Loader2 aria-hidden="true" className="animate-spin" />
      <AlertDescription>
        {state === "checking" ? t.gate.checking : t.gate.redirecting}{" "}
        {state === "denied" && (
          <Link href={`/${locale}/consent/`} className="font-medium text-primary underline">
            {t.gate.link}
          </Link>
        )}
      </AlertDescription>
    </Alert>
  );
}
