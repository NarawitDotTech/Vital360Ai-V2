import type { Tier } from "@/inference/types";

/** Icon shapes differ per tier so the tier is never conveyed by color alone. */
export function TierIcon({ tier, size = 56 }: { tier: Tier; size?: number }) {
  const common = { width: size, height: size, viewBox: "0 0 48 48", "aria-hidden": true, focusable: false } as const;
  switch (tier) {
    case "green":
      return (
        <svg {...common}>
          <circle cx="24" cy="24" r="21" fill="none" stroke="currentColor" strokeWidth="4" />
          <path d="M13 25l8 8 14-17" fill="none" stroke="currentColor" strokeWidth="5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      );
    case "amber":
      return (
        <svg {...common}>
          <path d="M24 5L45 41H3z" fill="none" stroke="currentColor" strokeWidth="4" strokeLinejoin="round" />
          <path d="M24 18v12" stroke="currentColor" strokeWidth="5" strokeLinecap="round" />
          <circle cx="24" cy="35" r="2.6" fill="currentColor" />
        </svg>
      );
    case "red":
      return (
        <svg {...common}>
          <path d="M15 4h18l11 11v18L33 44H15L4 33V15z" fill="none" stroke="currentColor" strokeWidth="4" strokeLinejoin="round" />
          <path d="M16 16l16 16M32 16L16 32" stroke="currentColor" strokeWidth="5" strokeLinecap="round" />
        </svg>
      );
    default:
      return (
        <svg {...common}>
          <circle cx="24" cy="24" r="21" fill="none" stroke="currentColor" strokeWidth="4" strokeDasharray="6 4" />
          <path d="M17 18c0-4 3-6 7-6s7 2 7 6c0 5-7 5-7 10" fill="none" stroke="currentColor" strokeWidth="4.5" strokeLinecap="round" />
          <circle cx="24" cy="35" r="2.6" fill="currentColor" />
        </svg>
      );
  }
}
