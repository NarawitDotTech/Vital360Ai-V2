/** Decorative motion widgets for the capture screens (aria-hidden; the text status next to them carries the meaning). */

export function ScanBar() {
  return <div aria-hidden="true" className="scan-bar" />;
}

export function Waveform() {
  return (
    <div aria-hidden="true" className="wave">
      {Array.from({ length: 14 }, (_, i) => (
        <span key={i} style={{ animationDelay: `${((i * 7) % 11) * -90}ms`, animationDuration: `${800 + ((i * 5) % 7) * 70}ms` }} />
      ))}
    </div>
  );
}

/** Smooth countdown ring; `fraction` 0..1 elapsed, animated linearly between 1 s ticks. */
export function CountdownRing({ fraction, label }: { fraction: number; label: string }) {
  const r = 26;
  const c = 2 * Math.PI * r;
  return (
    <div aria-hidden="true" className="relative size-16 shrink-0">
      <svg viewBox="0 0 64 64" className="size-full -rotate-90">
        <circle cx="32" cy="32" r={r} fill="none" stroke="var(--muted)" strokeWidth="6" />
        <circle
          cx="32" cy="32" r={r} fill="none" stroke="var(--primary)" strokeWidth="6" strokeLinecap="round"
          strokeDasharray={c} strokeDashoffset={c * (1 - fraction)} style={{ transition: "stroke-dashoffset 1s linear" }}
        />
      </svg>
      <span className="absolute inset-0 flex items-center justify-center text-base font-semibold tabular-nums">{label}</span>
    </div>
  );
}
