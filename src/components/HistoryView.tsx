"use client";

import { useEffect, useState } from "react";
import { bcp47, format } from "@/i18n";
import { clearHistory, deleteHistory, listHistory, type HistoryEntry } from "@/storage/db";
import { useI18n } from "./I18nProvider";
import { TierIcon } from "./TierBadge";

export function HistoryView() {
  const { locale, t } = useI18n();
  const [items, setItems] = useState<HistoryEntry[] | null>(null);
  const [error, setError] = useState(false);
  const [confirmAll, setConfirmAll] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  const [version, setVersion] = useState(0);
  const reload = () => setVersion((v) => v + 1);

  useEffect(() => {
    let cancelled = false;
    listHistory().then(
      (list) => {
        if (cancelled) return;
        setItems(list);
        setError(false);
      },
      () => {
        if (cancelled) return;
        setError(true);
        setItems([]);
      },
    );
    return () => {
      cancelled = true;
    };
  }, [version]);

  const fmtDate = (iso: string) =>
    new Intl.DateTimeFormat(bcp47[locale], { dateStyle: "medium", timeStyle: "short" }).format(new Date(iso));

  async function removeOne(id: string) {
    try {
      await deleteHistory(id);
      setNotice(t.history.deleted);
      reload();
    } catch {
      setError(true);
    }
  }
  async function removeAll() {
    try {
      await clearHistory();
      setConfirmAll(false);
      setNotice(t.history.deleted);
      reload();
    } catch {
      setError(true);
    }
  }

  return (
    <div className="stack">
      <h1>{t.history.title}</h1>
      <p className="text-muted">{t.history.note}</p>
      {error && (
        <p role="alert" className="notice notice-danger">
          {t.history.loadFailed}
        </p>
      )}
      {notice && (
        <p role="status" className="notice">
          {notice}
        </p>
      )}
      {items === null ? (
        <p role="status">{t.common.loading}</p>
      ) : items.length === 0 ? (
        <p className="card">{t.history.empty}</p>
      ) : (
        <>
          <ul className="stack-sm">
            {items.map((e) => {
              const tier = t.tiers[e.tier];
              const what = t.results.modality[e.modality];
              const date = fmtDate(e.createdAt);
              return (
                <li key={e.id} className={`tier tier-${e.tier} !border-2`}>
                  <div className="flex items-center gap-3">
                    <TierIcon tier={e.tier} size={40} />
                    <div className="flex-1">
                      <p className="font-bold">
                        {what}: {tier.word}
                      </p>
                      <p className="text-base">{date}</p>
                      {e.breathsPerMin !== undefined && (
                        <p className="text-base">{format(t.results.breaths, { n: Math.round(e.breathsPerMin) })}</p>
                      )}
                    </div>
                  </div>
                  <button
                    type="button"
                    className="btn btn-secondary mt-3 !bg-white"
                    onClick={() => removeOne(e.id)}
                    aria-label={format(t.history.deleteOneLabel, { what, date })}
                  >
                    {t.history.deleteOne}
                  </button>
                </li>
              );
            })}
          </ul>
          {confirmAll ? (
            <div className="stack-sm card">
              <button type="button" className="btn btn-danger" onClick={removeAll}>
                {t.history.confirmAll}
              </button>
              <button type="button" className="btn btn-secondary" onClick={() => setConfirmAll(false)}>
                {t.history.keep}
              </button>
            </div>
          ) : (
            <button type="button" className="btn btn-secondary" onClick={() => setConfirmAll(true)}>
              {t.history.deleteAll}
            </button>
          )}
        </>
      )}
    </div>
  );
}
