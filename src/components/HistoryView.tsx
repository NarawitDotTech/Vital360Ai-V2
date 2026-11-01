"use client";

import { CircleCheck, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { bcp47, format } from "@/i18n";
import {
  clearHistory,
  deleteHistory,
  listHistory,
  type HistoryEntry,
} from "@/storage/db";
import { Disclaimer } from "./Disclaimer";
import { useI18n } from "./I18nProvider";
import { isMockModel, MockNotice } from "./MockNotice";
import { canNameMatch, className } from "./names";
import { PageTitle } from "./PageTitle";
import { TierBadge } from "./TierBadge";
import { UrgentGuide } from "./UrgentGuide";

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
    new Intl.DateTimeFormat(bcp47[locale], {
      dateStyle: "medium",
      timeStyle: "short",
    }).format(new Date(iso));

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
    <div className="flex flex-col gap-5">
      <PageTitle>{t.history.title}</PageTitle>
      <p className="text-muted-foreground">{t.history.note}</p>
      <Disclaimer />
      <UrgentGuide />
      {error && (
        <Alert variant="destructive" role="alert">
          <AlertDescription>{t.history.loadFailed}</AlertDescription>
        </Alert>
      )}
      {notice && (
        <Alert variant="success" role="status">
          <CircleCheck aria-hidden="true" />
          <AlertDescription>{notice}</AlertDescription>
        </Alert>
      )}
      {items === null ? (
        <p role="status">{t.common.loading}</p>
      ) : items.length === 0 ? (
        <Card size="sm">
          <CardContent>
            <p>{t.history.empty}</p>
          </CardContent>
        </Card>
      ) : (
        <>
          <ul className="flex flex-col gap-3">
            {items.map((e) => {
              const what = t.results.modality[e.modality];
              const date = fmtDate(e.createdAt);
              return (
                <li key={e.id}>
                  <Card size="sm" data-tier={e.tier}>
                    <CardContent className="gap-3">
                      <div className="flex flex-wrap items-start justify-between gap-x-3 gap-y-2">
                        <div className="min-w-0">
                          <p className="font-semibold">{what}</p>
                          <p className="text-sm text-muted-foreground">
                            {date}
                          </p>
                          {canNameMatch(e, t.results) && (
                            <p className="text-sm">
                              {format(t.results.similarTo, {
                                name:
                                  className(t.results, e.modality, e.label) ??
                                  e.label,
                                pct: Math.round((e.probs[e.label] ?? 0) * 100),
                              })}
                            </p>
                          )}
                          {e.breathsPerMin !== undefined && (
                            <p className="text-sm">
                              {format(t.results.breaths, {
                                n: Math.round(e.breathsPerMin),
                              })}
                            </p>
                          )}
                        </div>
                        <TierBadge tier={e.tier} />
                      </div>
                      {isMockModel(e.modelVersion) && <MockNotice />}
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        className="self-start"
                        onClick={() => removeOne(e.id)}
                        aria-label={format(t.history.deleteOneLabel, {
                          what,
                          date,
                        })}
                      >
                        <Trash2 aria-hidden="true" />
                        {t.history.deleteOne}
                      </Button>
                    </CardContent>
                  </Card>
                </li>
              );
            })}
          </ul>
          {confirmAll ? (
            <Card size="sm">
              <CardContent className="gap-3">
                <Button
                  type="button"
                  variant="destructive"
                  size="lg"
                  onClick={removeAll}
                >
                  <Trash2 aria-hidden="true" />
                  {t.history.confirmAll}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="lg"
                  onClick={() => setConfirmAll(false)}
                >
                  {t.history.keep}
                </Button>
              </CardContent>
            </Card>
          ) : (
            <Button
              type="button"
              variant="outline"
              size="lg"
              className="self-start"
              onClick={() => setConfirmAll(true)}
            >
              <Trash2 aria-hidden="true" />
              {t.history.deleteAll}
            </Button>
          )}
        </>
      )}
    </div>
  );
}
