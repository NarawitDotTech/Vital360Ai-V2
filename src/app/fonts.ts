import { Noto_Sans_Thai, Roboto } from "next/font/google";

// Roboto for Latin text, Noto Sans Thai for Thai glyphs (Roboto has none). Self-hosted at build time.
export const roboto = Roboto({ subsets: ["latin"], variable: "--font-roboto", display: "swap" });
export const notoSansThai = Noto_Sans_Thai({ subsets: ["thai"], variable: "--font-noto-thai", display: "swap" });

export const fontVars = `${roboto.variable} ${notoSansThai.variable}`;
