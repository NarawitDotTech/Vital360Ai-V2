import type { Metadata, Viewport } from "next";
import "../globals.css";

export const metadata: Metadata = {
  title: "Vitals 360 AI",
  description: "Vitals 360 AI",
  manifest: "/manifest.webmanifest",
  icons: { icon: "/favicon-32.png", apple: "/apple-touch-icon.png" },
};
export const viewport: Viewport = { themeColor: "#0b6e72", width: "device-width", initialScale: 1 };

// "/" has no locale, so it is its own root layout (the locale layout is the root for everything else).
export default function RootRedirectLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="th">
      <body>{children}</body>
    </html>
  );
}
