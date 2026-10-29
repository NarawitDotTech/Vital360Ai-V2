"use client";

import Link from "next/link";
import { useState } from "react";
import { deleteAllData } from "@/storage/db";
import { useI18n } from "./I18nProvider";

export function DeleteDataView() {
  const { locale, t } = useI18n();
  const [step, setStep] = useState<"idle" | "confirm" | "busy" | "done" | "error">("idle");

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
      <div className="stack">
        <h1>{t.deleteData.title}</h1>
        <p role="status" className="notice text-xl font-bold" data-testid="delete-done">
          ✓ {t.deleteData.done}
        </p>
        <p>{t.deleteData.doneNote}</p>
        <Link href={`/${locale}/`} className="btn btn-primary">
          {t.common.home}
        </Link>
      </div>
    );
  }

  return (
    <div className="stack">
      <h1>{t.deleteData.title}</h1>
      <p>{t.deleteData.intro}</p>
      <ul className="list-disc space-y-1 pl-6">
        {t.deleteData.lists.map((l) => (
          <li key={l}>{l}</li>
        ))}
      </ul>
      <p className="notice">{t.deleteData.account}</p>
      {step === "error" && (
        <p role="alert" className="notice notice-danger">
          {t.deleteData.failed}
        </p>
      )}
      {step === "confirm" || step === "busy" ? (
        <div className="card stack-sm" role="group" aria-labelledby="del-confirm">
          <h2 id="del-confirm">{t.deleteData.confirmTitle}</h2>
          <button type="button" className="btn btn-danger" onClick={wipe} disabled={step === "busy"}>
            {t.deleteData.confirm}
          </button>
          <button type="button" className="btn btn-secondary" onClick={() => setStep("idle")} disabled={step === "busy"}>
            {t.deleteData.keep}
          </button>
        </div>
      ) : (
        <button type="button" className="btn btn-danger btn-big" onClick={() => setStep("confirm")}>
          {t.deleteData.button}
        </button>
      )}
    </div>
  );
}
