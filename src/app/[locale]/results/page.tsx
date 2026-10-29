import type { Metadata } from "next";
import { ConsentGate } from "@/components/ConsentGate";
import { ResultsView } from "@/components/ResultsView";
import { getDict } from "@/i18n";

export async function generateMetadata({ params }: { params: Promise<{ locale: string }> }): Promise<Metadata> {
  return { title: getDict((await params).locale).results.title };
}

export default function Page() {
  return (
    <ConsentGate>
      <ResultsView />
    </ConsentGate>
  );
}
