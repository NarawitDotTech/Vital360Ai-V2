import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";
import { fontVars } from "./fonts";

export const metadata: Metadata = { title: "ไม่พบหน้านี้ · Page not found — Vitals 360 AI" };

const LINK =
  "inline-flex min-h-12 items-center justify-center rounded-lg px-5 text-lg font-semibold no-underline " +
  "outline-none focus-visible:ring-[3px] focus-visible:ring-ring";

// Unmatched URLs have no locale, so this page is bilingual and brings its own <html>.
export default function GlobalNotFound() {
  return (
    <html lang="th" className={fontVars}>
      <body>
        <main className="mx-auto flex min-h-dvh max-w-md flex-col justify-center gap-5 px-4 py-10 text-center">
          <p aria-hidden="true" className="text-6xl font-semibold text-primary">
            404
          </p>
          <h1 className="text-2xl leading-normal font-semibold">ไม่พบหน้านี้</h1>
          <p className="text-muted-foreground" lang="en">
            This page could not be found.
          </p>
          <div className="flex flex-col gap-3">
            <Link href="/th/" className={`${LINK} bg-primary text-primary-foreground`}>
              กลับหน้าแรก
            </Link>
            <Link href="/en/" lang="en" className={`${LINK} border bg-card text-foreground`}>
              Back to home
            </Link>
          </div>
        </main>
      </body>
    </html>
  );
}
