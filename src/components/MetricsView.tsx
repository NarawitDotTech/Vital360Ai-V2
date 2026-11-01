"use client";

import { Download, FlaskConical, ImageDown } from "lucide-react";
import { useCallback, useEffect, useRef } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { format } from "@/i18n";
import {
  MODEL_KEYS,
  formatBytes,
  metricsToCsv,
  type MetricsFile,
  type ModelKey,
} from "@/eval/metrics";
import { useI18n } from "./I18nProvider";
import { PageTitle } from "./PageTitle";

// Colour-blind-safe series colours (Okabe-Ito); every bar also has a printed value and the legend has names.
const SERIES: Record<ModelKey, string> = {
  skin: "#0072B2",
  audio: "#E69F00",
  resp: "#009E73",
  fusion: "#7a2e6b",
};
const METRICS = ["accuracy", "sensitivity", "specificity", "f1"] as const;

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

function download(name: string, blob: Blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function drawChart(
  canvas: HTMLCanvasElement,
  file: MetricsFile,
  names: Record<ModelKey, string>,
  title: string,
) {
  const W = 960;
  const H = 540;
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const keys = MODEL_KEYS.filter((k) => file.models[k]);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = "#12202a";
  ctx.font = "bold 26px sans-serif";
  ctx.fillText(title, 24, 40);
  if (file.isSample) {
    ctx.fillStyle = "#991b1b";
    ctx.font = "bold 22px sans-serif";
    ctx.fillText("SAMPLE DATA, not measured", 24, 72);
  }
  const left = 60;
  const top = 100;
  const bottom = H - 90;
  const right = W - 24;
  ctx.strokeStyle = "#9fb4b8";
  ctx.fillStyle = "#3d4e5a";
  ctx.font = "16px sans-serif";
  for (let i = 0; i <= 5; i++) {
    const v = i / 5;
    const y = bottom - v * (bottom - top);
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(right, y);
    ctx.stroke();
    ctx.fillText(`${Math.round(v * 100)}%`, 10, y + 5);
  }
  const groupW = (right - left) / METRICS.length;
  const barW = Math.min(46, (groupW - 24) / Math.max(1, keys.length));
  METRICS.forEach((metric, gi) => {
    const gx = left + gi * groupW + (groupW - barW * keys.length) / 2;
    keys.forEach((k, ki) => {
      const v = file.models[k]![metric];
      const h = v * (bottom - top);
      const x = gx + ki * barW;
      ctx.fillStyle = SERIES[k];
      ctx.fillRect(x + 2, bottom - h, barW - 4, h);
      ctx.fillStyle = "#12202a";
      ctx.font = "bold 13px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(`${Math.round(v * 100)}`, x + barW / 2, bottom - h - 6);
      ctx.textAlign = "left";
    });
    ctx.fillStyle = "#12202a";
    ctx.font = "bold 17px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(metric, left + gi * groupW + groupW / 2, bottom + 26);
    ctx.textAlign = "left";
  });
  let lx = left;
  ctx.font = "16px sans-serif";
  keys.forEach((k) => {
    ctx.fillStyle = SERIES[k];
    ctx.fillRect(lx, H - 40, 18, 18);
    ctx.fillStyle = "#12202a";
    ctx.fillText(names[k], lx + 26, H - 25);
    lx += 40 + ctx.measureText(names[k]).width;
  });
}

function FileSection({ file }: { file: MetricsFile }) {
  const { t } = useI18n();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const keys = MODEL_KEYS.filter((k) => file.models[k]);
  const names = t.dashboard.models as Record<ModelKey, string>;

  useEffect(() => {
    if (canvasRef.current)
      drawChart(canvasRef.current, file, names, t.dashboard.chartTitle);
  }, [file, names, t.dashboard.chartTitle]);

  const exportCsv = useCallback(() => {
    download(
      file.file.replace(/\.json$/, "") + ".csv",
      new Blob([metricsToCsv(file)], { type: "text/csv;charset=utf-8" }),
    );
  }, [file]);
  const exportPng = useCallback(() => {
    const c = document.createElement("canvas");
    drawChart(c, file, names, t.dashboard.chartTitle);
    c.toBlob(
      (b) => b && download(file.file.replace(/\.json$/, "") + ".png", b),
      "image/png",
    );
  }, [file, names, t.dashboard.chartTitle]);

  return (
    <section className="flex flex-col gap-5" aria-labelledby={`f-${file.file}`}>
      <h2 id={`f-${file.file}`} className="text-xl font-semibold">
        {format(t.dashboard.fileLabel, { file: file.file })}
      </h2>
      {file.isSample && (
        <Alert variant="destructive" role="note" data-testid="sample-banner">
          <FlaskConical aria-hidden="true" />
          <AlertTitle className="text-lg font-semibold">
            {t.dashboard.sampleBanner}
          </AlertTitle>
          <AlertDescription>{t.dashboard.sampleDetail}</AlertDescription>
        </Alert>
      )}
      {file.description && <p>{file.description}</p>}
      {file.generatedAt && (
        <p className="text-sm text-muted-foreground">
          {format(t.dashboard.generatedAt, { date: file.generatedAt })}
        </p>
      )}

      <Card size="sm">
        <CardHeader>
          <CardTitle>
            <h3>{t.dashboard.compareTitle}</h3>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableCaption className="sr-only">
              {t.dashboard.compareTitle}
            </TableCaption>
            <TableHeader>
              <TableRow>
                <TableHead scope="col">{t.dashboard.model}</TableHead>
                <TableHead scope="col" className="text-right">
                  {t.dashboard.accuracy}
                </TableHead>
                <TableHead scope="col" className="text-right">
                  {t.dashboard.sensitivity}
                </TableHead>
                <TableHead scope="col" className="text-right">
                  {t.dashboard.specificity}
                </TableHead>
                <TableHead scope="col" className="text-right">
                  {t.dashboard.f1}
                </TableHead>
                <TableHead scope="col" className="text-right">
                  {t.dashboard.latency}
                </TableHead>
                <TableHead scope="col" className="text-right">
                  {t.dashboard.size}
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {keys.map((k) => {
                const m = file.models[k]!;
                return (
                  <TableRow
                    key={k}
                    className={k === "fusion" ? "font-semibold" : undefined}
                  >
                    <TableHead scope="row">{names[k]}</TableHead>
                    <TableCell className="text-right">
                      {pct(m.accuracy)}
                    </TableCell>
                    <TableCell className="text-right">
                      {pct(m.sensitivity)}
                    </TableCell>
                    <TableCell className="text-right">
                      {pct(m.specificity)}
                    </TableCell>
                    <TableCell className="text-right">{pct(m.f1)}</TableCell>
                    <TableCell className="text-right">{m.latencyMs}</TableCell>
                    <TableCell className="text-right">
                      {formatBytes(m.sizeBytes)}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Card size="sm">
        <CardHeader>
          <CardTitle>
            <h3>{t.dashboard.chartTitle}</h3>
          </CardTitle>
        </CardHeader>
        <CardContent className="gap-3">
          <canvas
            ref={canvasRef}
            role="img"
            aria-label={t.dashboard.chartAlt}
            className="w-full rounded-lg border"
          />
          <div className="grid gap-3 sm:grid-cols-2">
            <Button
              type="button"
              variant="outline"
              size="lg"
              onClick={exportCsv}
            >
              <Download aria-hidden="true" />
              {t.dashboard.exportCsv}
            </Button>
            <Button
              type="button"
              variant="outline"
              size="lg"
              onClick={exportPng}
            >
              <ImageDown aria-hidden="true" />
              {t.dashboard.exportPng}
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card size="sm">
        <CardHeader>
          <CardTitle>
            <h3>{t.dashboard.perClassTitle}</h3>
          </CardTitle>
        </CardHeader>
        <CardContent className="gap-5">
          {keys.map((k) => {
            const m = file.models[k]!;
            return (
              <Table key={k}>
                <TableCaption className="mt-0 mb-2 text-left font-semibold text-foreground caption-top">
                  {names[k]}
                </TableCaption>
                <TableHeader>
                  <TableRow>
                    <TableHead scope="col">{t.dashboard.classCol}</TableHead>
                    <TableHead scope="col" className="text-right">
                      {t.dashboard.sensitivity}
                    </TableHead>
                    <TableHead scope="col" className="text-right">
                      {t.dashboard.specificity}
                    </TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {m.classes.map((c) => (
                    <TableRow key={c}>
                      <TableHead scope="row">{c}</TableHead>
                      <TableCell className="text-right">
                        {pct(m.perClass[c].sensitivity)}
                      </TableCell>
                      <TableCell className="text-right">
                        {pct(m.perClass[c].specificity)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            );
          })}
        </CardContent>
      </Card>

      <Card size="sm">
        <CardHeader>
          <CardTitle>
            <h3>{t.dashboard.confusionTitle}</h3>
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            {t.dashboard.confusionHint}
          </p>
        </CardHeader>
        <CardContent className="gap-5">
          {keys.map((k) => {
            const m = file.models[k]!;
            const max = Math.max(1, ...m.confusionMatrix.flat());
            return (
              <Table key={k}>
                <TableCaption className="mt-0 mb-2 text-left font-semibold text-foreground caption-top">
                  {names[k]}
                </TableCaption>
                <TableHeader>
                  <TableRow>
                    <TableHead scope="col">
                      {t.dashboard.actual} \ {t.dashboard.predicted}
                    </TableHead>
                    {m.classes.map((c) => (
                      <TableHead key={c} scope="col" className="text-right">
                        {c}
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {m.classes.map((rowName, r) => (
                    <TableRow key={rowName}>
                      <TableHead scope="row">{rowName}</TableHead>
                      {m.confusionMatrix[r].map((v, c) => (
                        <TableCell
                          key={c}
                          style={{
                            background: `rgba(11,110,114,${(0.08 + (0.5 * v) / max).toFixed(2)})`,
                          }}
                          className={
                            r === c ? "text-right font-semibold" : "text-right"
                          }
                        >
                          {v}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            );
          })}
        </CardContent>
      </Card>

      <Card size="sm">
        <CardHeader>
          <CardTitle>
            <h3>{t.dashboard.testSet}</h3>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="list-disc space-y-1.5 pl-5">
            {keys.map((k) => (
              <li key={k}>
                <strong className="font-semibold">{names[k]}:</strong>{" "}
                {file.models[k]!.testSetDescription}
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
    </section>
  );
}

export function MetricsView({ files }: { files: MetricsFile[] }) {
  const { t } = useI18n();
  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-2">
        <PageTitle>{t.dashboard.title}</PageTitle>
        <p className="text-muted-foreground">{t.dashboard.intro}</p>
      </div>
      {files.length === 0 && (
        <Card size="sm">
          <CardContent>
            <p>{t.dashboard.empty}</p>
          </CardContent>
        </Card>
      )}
      {files.map((f) => (
        <FileSection key={f.file} file={f} />
      ))}
    </div>
  );
}
