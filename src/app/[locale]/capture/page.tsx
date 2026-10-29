import type { Metadata } from "next";
import { CaptureHub } from "@/components/CaptureHub";
import { ConsentGate } from "@/components/ConsentGate";
import { getDict } from "@/i18n";

export async function generateMetadata({ params }: { params: Promise<{ locale: string }> }): Promise<Metadata> {
  return { title: getDict((await params).locale).capture.title };
}

export default function CapturePage() {
  return (
    <ConsentGate>
      <CaptureHub />
    </ConsentGate>
  );
}
