import type { Metadata } from "next";
import { MetricsView } from "@/components/MetricsView";
import { loadMetricFiles } from "@/eval/load";
import { getDict } from "@/i18n";

export async function generateMetadata({ params }: { params: Promise<{ locale: string }> }): Promise<Metadata> {
  return { title: getDict((await params).locale).dashboard.title };
}

export default function DashboardPage() {
  // Read at build time (static export); rebuild after adding files to eval/metrics/.
  return <MetricsView files={loadMetricFiles()} />;
}
