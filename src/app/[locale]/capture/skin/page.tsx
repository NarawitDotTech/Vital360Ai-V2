import type { Metadata } from "next";
import { ConsentGate } from "@/components/ConsentGate";
import { SkinCaptureLoader } from "@/components/capture/Loaders";
import { getDict } from "@/i18n";

export async function generateMetadata({ params }: { params: Promise<{ locale: string }> }): Promise<Metadata> {
  return { title: getDict((await params).locale).skin.title };
}

export default function SkinPage() {
  return (
    <ConsentGate>
      <SkinCaptureLoader />
    </ConsentGate>
  );
}
