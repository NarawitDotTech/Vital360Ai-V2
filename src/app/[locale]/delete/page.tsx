import type { Metadata } from "next";

import { DeleteDataView } from "@/components/DeleteDataView";
import { getDict } from "@/i18n";

export async function generateMetadata({ params }: { params: Promise<{ locale: string }> }): Promise<Metadata> {
  return { title: getDict((await params).locale).deleteData.title };
}

export default function Page() {
  return <DeleteDataView />;
}
