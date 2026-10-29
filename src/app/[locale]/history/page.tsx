import type { Metadata } from "next";

import { HistoryView } from "@/components/HistoryView";
import { getDict } from "@/i18n";

export async function generateMetadata({ params }: { params: Promise<{ locale: string }> }): Promise<Metadata> {
  return { title: getDict((await params).locale).history.title };
}

export default function Page() {
  return <HistoryView />;
}
