"use client";

import { CircleCheck, Info, Trash2 } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { deleteAllData } from "@/storage/db";
import { useI18n } from "./I18nProvider";
import { PageTitle } from "./PageTitle";

export function DeleteDataView() {
  const { locale, t } = useI18n();
  const [step, setStep] = useState<
    "idle" | "confirm" | "busy" | "done" | "error"
  >("idle");

  async function wipe() {
    setStep("busy");
    try {
      await deleteAllData();
      setStep("done");
    } catch {
      setStep("error");
    }
  }

  if (step === "done") {
    return (
      <div className="flex flex-col gap-5">
        <PageTitle>{t.deleteData.title}</PageTitle>
        <Alert variant="success" role="status" data-testid="delete-done">
          <CircleCheck aria-hidden="true" />
          <AlertTitle className="font-semibold">{t.deleteData.done}</AlertTitle>
        </Alert>
        <p>{t.deleteData.doneNote}</p>
        <Button asChild size="lg" className="w-full">
          <Link href={`/${locale}/`}>{t.common.home}</Link>
        </Button>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-5">
      <PageTitle>{t.deleteData.title}</PageTitle>
      <p>{t.deleteData.intro}</p>
      <Card size="sm">
        <CardContent>
          <ul className="list-disc space-y-1.5 pl-5">
            {t.deleteData.lists.map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
        </CardContent>
      </Card>
      <Alert role="note">
        <Info aria-hidden="true" />
        <AlertDescription>{t.deleteData.account}</AlertDescription>
      </Alert>
      {step === "error" && (
        <Alert variant="destructive" role="alert">
          <AlertDescription>{t.deleteData.failed}</AlertDescription>
        </Alert>
      )}
      {step === "confirm" || step === "busy" ? (
        <Card role="group" aria-labelledby="del-confirm">
          <CardHeader>
            <CardTitle>
              <h2 id="del-confirm">{t.deleteData.confirmTitle}</h2>
            </CardTitle>
          </CardHeader>
          <CardContent className="gap-3">
            <Button
              type="button"
              variant="destructive"
              size="lg"
              className="w-full"
              onClick={wipe}
              disabled={step === "busy"}
            >
              <Trash2 aria-hidden="true" />
              {t.deleteData.confirm}
            </Button>
            <Button
              type="button"
              variant="outline"
              size="lg"
              onClick={() => setStep("idle")}
              disabled={step === "busy"}
            >
              {t.deleteData.keep}
            </Button>
          </CardContent>
        </Card>
      ) : (
        <Button
          type="button"
          variant="destructive"
          size="lg"
          className="w-full"
          onClick={() => setStep("confirm")}
        >
          <Trash2 aria-hidden="true" />
          {t.deleteData.button}
        </Button>
      )}
    </div>
  );
}
