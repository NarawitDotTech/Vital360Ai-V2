"use client";

import Link from "next/link";
import { useEffect } from "react";
import { PageTitle } from "@/components/PageTitle";
import { Button } from "@/components/ui/button";

// Static export has no server redirect, so redirect on the client and keep visible links as fallback.
export default function RootPage() {
  useEffect(() => {
    window.location.replace("/th/");
  }, []);
  return (
    <main className="mx-auto flex w-full max-w-2xl flex-col gap-5 px-4 pt-6 pb-10">
      <PageTitle>Vitals 360 AI</PageTitle>
      <p role="status">กำลังพาไปหน้าภาษาไทย… / Taking you to the Thai page…</p>
      <Button asChild size="lg" className="w-full">
        <Link href="/th/" lang="th">
          ไปที่ Vitals 360 AI (ภาษาไทย)
        </Link>
      </Button>
      <Button asChild variant="outline" size="lg">
        <Link href="/en/" lang="en">
          Go to Vitals 360 AI (English)
        </Link>
      </Button>
      <noscript>
        <p>JavaScript is needed to use this app. กรุณาเปิดใช้ JavaScript</p>
      </noscript>
    </main>
  );
}
