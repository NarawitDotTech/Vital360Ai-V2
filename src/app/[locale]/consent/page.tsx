import type { Metadata } from "next";
import { getDict } from "@/i18n";
import { ConsentForm } from "@/components/ConsentForm";

export async function generateMetadata({ params }: { params: Promise<{ locale: string }> }): Promise<Metadata> {
  return { title: getDict((await params).locale).consent.title };
}

export default function ConsentPage() {
  return <ConsentForm />;
}
