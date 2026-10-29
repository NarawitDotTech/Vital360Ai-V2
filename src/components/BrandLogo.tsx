import { readFileSync } from "node:fs";
import { join } from "node:path";

/**
 * Inlines public/icon.svg (already sanitised by scripts/icons.mjs) at build time.
 * Class names are prefixed and ids dropped so several inline copies cannot collide with page CSS.
 */
function inlineSvg(): string {
  const raw = readFileSync(join(process.cwd(), "public", "icon.svg"), "utf8");
  return raw
    .replace(/<\?xml[^>]*\?>/g, "")
    .replace(/\sid="[^"]*"/g, "")
    .replace(/\.s(\d+)\s*\{/g, ".v360-s$1 {")
    .replace(/class="s(\d+)"/g, 'class="v360-s$1"')
    .replace(/<svg\b([^>]*)>/, (_m, attrs: string) => {
      const cleaned = attrs.replace(/\s(width|height)="[^"]*"/g, "");
      return `<svg${cleaned} width="48" height="48" aria-hidden="true" focusable="false">`;
    });
}

export function BrandLogo() {
  return <span className="inline-flex shrink-0 rounded-xl overflow-hidden" dangerouslySetInnerHTML={{ __html: inlineSvg() }} />;
}
