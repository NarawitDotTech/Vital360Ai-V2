import type { Metadata } from "next";
import { ConsentGate } from "@/components/ConsentGate";
import { AudioCaptureLoader } from "@/components/capture/Loaders";
import { getDict } from "@/i18n";

export async function generateMetadata({ params }: { params: Promise<{ locale: string }> }): Promise<Metadata> {
  return { title: getDict((await params).locale).audio.title };
}

export default function AudioPage() {
  return (
    <ConsentGate>
      <AudioCaptureLoader />
    </ConsentGate>
  );
}
