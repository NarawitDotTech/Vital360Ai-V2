"use client";

import { useEffect } from "react";

// Static export has no server redirect, so redirect on the client and keep visible links as fallback.
export default function RootPage() {
  useEffect(() => {
    window.location.replace("/th/");
  }, []);
  return (
    <main className="page stack">
      <h1>Vitals 360 AI</h1>
      <p role="status">กำลังพาไปหน้าภาษาไทย… / Taking you to the Thai page…</p>
      <a className="btn btn-primary" href="/th/" lang="th">
        ไปที่ Vitals 360 AI (ภาษาไทย)
      </a>
      <a className="btn btn-secondary" href="/en/" lang="en">
        Go to Vitals 360 AI (English)
      </a>
      <noscript>
        <p>JavaScript is needed to use this app. กรุณาเปิดใช้ JavaScript</p>
      </noscript>
    </main>
  );
}
