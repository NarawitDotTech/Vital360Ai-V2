// Generates all icons from assets/icon.svg. Run: npm run icons
// 1. sanitise the SVG (no script, foreignObject, event handlers, external references) -> public/icon.svg
// 2. favicon 32, apple-touch-icon 180, PWA 192/512, maskable 512 (safe-zone padding), OG image 1200x630.
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import sharp from "sharp";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const SRC = join(root, "assets", "icon.svg");
const OUT = join(root, "public");
const BRAND_DARK = "#0b6e72";
const BG = "#fdfdfd"; // same as the icon background so padding is seamless

export function sanitizeSvg(input) {
  let s = input;
  s = s.replace(/<\?xml[\s\S]*?\?>/g, "").replace(/<!DOCTYPE[\s\S]*?>/gi, "").replace(/<!--[\s\S]*?-->/g, "");
  s = s.replace(/<script\b[\s\S]*?<\/script\s*>/gi, "").replace(/<script\b[^>]*\/>/gi, "");
  s = s.replace(/<foreignObject\b[\s\S]*?<\/foreignObject\s*>/gi, "");
  s = s.replace(/<(iframe|object|embed|audio|video|link|meta|animate|set)\b[\s\S]*?(<\/\1\s*>|\/>)/gi, "");
  // event handler attributes: onclick="..." onload='...' onx=bare
  s = s.replace(/\s+on[a-z]+\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)/gi, "");
  // href / xlink:href that do not point at an in-document fragment
  s = s.replace(/\s+(?:xlink:)?href\s*=\s*("([^"]*)"|'([^']*)')/gi, (m, _q, a, b) => ((a ?? b ?? "").trim().startsWith("#") ? m : ""));
  // style blocks: drop @import and external url()
  s = s.replace(/<style\b([^>]*)>([\s\S]*?)<\/style\s*>/gi, (_m, attrs, css) => {
    const clean = css.replace(/@import[^;]*;?/gi, "").replace(/url\(\s*(['"]?)(?!#)[^)]*\1\s*\)/gi, "none");
    return `<style${attrs}>${clean}</style>`;
  });
  s = s.replace(/\s+style\s*=\s*("[^"]*"|'[^']*')/gi, (m) => (/url\(|@import|expression\(|javascript:/i.test(m) ? "" : m));
  s = s.replace(/javascript:/gi, "");
  return s.trim() + "\n";
}

async function png(svg, size, file, { pad = 0, background = BG } = {}) {
  const inner = Math.round(size * (1 - 2 * pad));
  const icon = await sharp(Buffer.from(svg), { density: 384 }).resize(inner, inner, { fit: "contain", background }).png().toBuffer();
  await sharp({ create: { width: size, height: size, channels: 4, background } })
    .composite([{ input: icon, left: Math.round((size - inner) / 2), top: Math.round((size - inner) / 2) }])
    .png({ compressionLevel: 9 })
    .toFile(join(OUT, file));
}

async function ogImage(svg) {
  const W = 1200;
  const H = 630;
  const icon = await sharp(Buffer.from(svg), { density: 384 }).resize(360, 360).png().toBuffer();
  const text = Buffer.from(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${W - 520}" height="360">
       <text x="0" y="170" font-family="Helvetica, Arial, sans-serif" font-size="84" font-weight="700" fill="${BRAND_DARK}">Vitals 360 AI</text>
       <text x="0" y="240" font-family="Helvetica, Arial, sans-serif" font-size="36" fill="#12202a">Private on-device health screening</text>
       <text x="0" y="290" font-family="Helvetica, Arial, sans-serif" font-size="28" fill="#3d4e5a">Skin, cough and breathing. Not a diagnosis.</text>
     </svg>`,
  );
  await sharp({ create: { width: W, height: H, channels: 4, background: BG } })
    .composite([
      { input: Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="14"><rect width="${W}" height="14" fill="#23c2c6"/></svg>`), left: 0, top: 0 },
      { input: icon, left: 100, top: 135 },
      { input: text, left: 500, top: 135 },
    ])
    .png({ compressionLevel: 9 })
    .toFile(join(OUT, "og.png"));
}

async function main() {
  await mkdir(OUT, { recursive: true });
  const svg = sanitizeSvg(await readFile(SRC, "utf8"));
  if (!/<svg\b/i.test(svg)) throw new Error("assets/icon.svg has no <svg> root");
  await writeFile(join(OUT, "icon.svg"), svg);
  await png(svg, 32, "favicon-32.png");
  await png(svg, 180, "apple-touch-icon.png");
  await png(svg, 192, "icon-192.png");
  await png(svg, 512, "icon-512.png");
  await png(svg, 512, "icon-maskable-512.png", { pad: 0.14 }); // content inside the central ~72%, the maskable safe zone is 80%
  await ogImage(svg);
  console.log("icons written to public/");
}

if (process.argv[1] === fileURLToPath(import.meta.url)) await main();
