"use client";

import { CircleCheck, CircleHelp, OctagonAlert, TriangleAlert, type LucideIcon } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { Tier } from "@/inference/types";
import { useI18n } from "./I18nProvider";

/** Icon shapes differ per tier (circle, triangle, octagon, dashed question) so the tier is never conveyed by color alone. */
const ICONS: Record<Tier, LucideIcon> = {
  green: CircleCheck,
  amber: TriangleAlert,
  red: OctagonAlert,
  uncertain: CircleHelp,
};

const TINTS: Record<Tier, string> = {
  green: "border-tier-green-border bg-tier-green-bg text-tier-green-fg",
  amber: "border-tier-amber-border bg-tier-amber-bg text-tier-amber-fg",
  red: "border-tier-red-border bg-tier-red-bg text-tier-red-fg",
  uncertain: "border-tier-uncertain-border bg-tier-uncertain-bg text-tier-uncertain-fg",
};

export function TierIcon({ tier, className }: { tier: Tier; className?: string }) {
  const Icon = ICONS[tier];
  return <Icon aria-hidden="true" className={className} />;
}

/** Soft tinted badge: icon + word. `size="lg"` is the headline badge on a result card. */
export function TierBadge({ tier, size = "md" }: { tier: Tier; size?: "md" | "lg" }) {
  const { t } = useI18n();
  return (
    <Badge variant="outline" className={cn(TINTS[tier], size === "lg" && "h-9 gap-2 px-4 text-lg [&>svg]:size-5!")} data-tier-badge={tier}>
      <TierIcon tier={tier} />
      {t.tiers[tier].word}
    </Badge>
  );
}
