"use client";

import { useCallback, useEffect, useRef } from "react";
import { format } from "@/i18n";
import { MODEL_KEYS, formatBytes, metricsToCsv, type MetricsFile, type ModelKey } from "@/eval/metrics";
import { useI18n } from "./I18nProvider";

// Colour-blind-safe series colours (Okabe-Ito); every bar also has a printed value and the legend has names.
const SERIES: Record<ModelKey, string> = { skin: "#0072B2", audio: "#E69F00", resp: "#009E73", fusion: "#7a2e6b" };
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

function drawChart(canvas: HTMLCanvasElement, file: MetricsFile, names: Record<ModelKey, string>, title: string) {
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
    if (canvasRef.current) drawChart(canvasRef.current, file, names, t.dashboard.chartTitle);
  }, [file, names, t.dashboard.chartTitle]);

  const exportCsv = useCallback(() => {
    download(file.file.replace(/\.json$/, "") + ".csv", new Blob([metricsToCsv(file)], { type: "text/csv;charset=utf-8" }));
  }, [file]);
  const exportPng = useCallback(() => {
    const c = document.createElement("canvas");
    drawChart(c, file, names, t.dashboard.chartTitle);
    c.toBlob((b) => b && download(file.file.replace(/\.json$/, "") + ".png", b), "image/png");
  }, [file, names, t.dashboard.chartTitle]);

  return (
    <section className="stack" aria-labelledby={`f-${file.file}`}>
      <h2 id={`f-${file.file}`}>{format(t.dashboard.fileLabel, { file: file.file })}</h2>
      {file.isSample && (
        <div role="note" className="notice notice-danger" data-testid="sample-banner">
          <p className="text-2xl font-extrabold">⚠ {t.dashboard.sampleBanner}</p>
          <p>{t.dashboard.sampleDetail}</p>
        </div>
      )}
      {file.description && <p>{file.description}</p>}
      {file.generatedAt && <p className="text-sm text-muted">{format(t.dashboard.generatedAt, { date: file.generatedAt })}</p>}

      <h3>{t.dashboard.compareTitle}</h3>
      <div className="table-scroll">
        <table className="data">
          <caption className="sr-only">{t.dashboard.compareTitle}</caption>
          <thead>
            <tr>
              <th scope="col">{t.dashboard.model}</th>
              <th scope="col">{t.dashboard.accuracy}</th>
              <th scope="col">{t.dashboard.sensitivity}</th>
              <th scope="col">{t.dashboard.specificity}</th>
              <th scope="col">{t.dashboard.f1}</th>
              <th scope="col">{t.dashboard.latency}</th>
              <th scope="col">{t.dashboard.size}</th>
            </tr>
          </thead>
          <tbody>
            {keys.map((k) => {
              const m = file.models[k]!;
              return (
                <tr key={k} className={k === "fusion" ? "font-bold" : undefined}>
                  <th scope="row">{names[k]}</th>
                  <td>{pct(m.accuracy)}</td>
                  <td>{pct(m.sensitivity)}</td>
                  <td>{pct(m.specificity)}</td>
                  <td>{pct(m.f1)}</td>
                  <td>{m.latencyMs}</td>
                  <td>{formatBytes(m.sizeBytes)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <h3>{t.dashboard.chartTitle}</h3>
      <canvas ref={canvasRef} role="img" aria-label={t.dashboard.chartAlt} className="w-full rounded-lg border-2 border-[var(--line)]" />
      <div className="grid gap-3 sm:grid-cols-2">
        <button type="button" className="btn btn-secondary" onClick={exportCsv}>
          {t.dashboard.exportCsv}
        </button>
        <button type="button" className="btn btn-secondary" onClick={exportPng}>
          {t.dashboard.exportPng}
        </button>
      </div>

      <h3>{t.dashboard.perClassTitle}</h3>
      {keys.map((k) => {
        const m = file.models[k]!;
        return (
          <div key={k} className="table-scroll">
            <table className="data">
              <caption className="text-left font-bold">{names[k]}</caption>
              <thead>
                <tr>
                  <th scope="col">{t.dashboard.classCol}</th>
                  <th scope="col">{t.dashboard.sensitivity}</th>
                  <th scope="col">{t.dashboard.specificity}</th>
                </tr>
              </thead>
              <tbody>
                {m.classes.map((c) => (
                  <tr key={c}>
                    <th scope="row">{c}</th>
                    <td>{pct(m.perClass[c].sensitivity)}</td>
                    <td>{pct(m.perClass[c].specificity)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      })}

      <h3>{t.dashboard.confusionTitle}</h3>
      <p className="text-sm text-muted">{t.dashboard.confusionHint}</p>
      {keys.map((k) => {
        const m = file.models[k]!;
        const max = Math.max(1, ...m.confusionMatrix.flat());
        return (
          <div key={k} className="table-scroll">
            <table className="data">
              <caption className="text-left font-bold">{names[k]}</caption>
              <thead>
                <tr>
                  <th scope="col">
                    {t.dashboard.actual} \ {t.dashboard.predicted}
                  </th>
                  {m.classes.map((c) => (
                    <th key={c} scope="col">
                      {c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {m.classes.map((rowName, r) => (
                  <tr key={rowName}>
                    <th scope="row">{rowName}</th>
                    {m.confusionMatrix[r].map((v, c) => (
                      <td
                        key={c}
                        style={{ background: `rgba(11,110,114,${(0.08 + (0.5 * v) / max).toFixed(2)})` }}
                        className={r === c ? "font-bold" : undefined}
                      >
                        {v}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      })}

      <h3>{t.dashboard.testSet}</h3>
      <ul className="list-disc space-y-1 pl-6">
        {keys.map((k) => (
          <li key={k}>
            <strong>{names[k]}:</strong> {file.models[k]!.testSetDescription}
          </li>
        ))}
      </ul>
    </section>
  );
}

export function MetricsView({ files }: { files: MetricsFile[] }) {
  const { t } = useI18n();
  return (
    <div className="stack">
      <h1>{t.dashboard.title}</h1>
      <p>{t.dashboard.intro}</p>
      {files.length === 0 && <p className="card">{t.dashboard.empty}</p>}
      {files.map((f) => (
        <FileSection key={f.file} file={f} />
      ))}
    </div>
  );
}
