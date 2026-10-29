import type { Metadata } from "next";
import { ConsentGate } from "@/components/ConsentGate";
import { RespCaptureLoader } from "@/components/capture/Loaders";
import { getDict } from "@/i18n";

export async function generateMetadata({ params }: { params: Promise<{ locale: string }> }): Promise<Metadata> {
  return { title: getDict((await params).locale).resp.title };
}

export default function RespPage() {
  return (
    <ConsentGate>
      <RespCaptureLoader />
    </ConsentGate>
  );
}
