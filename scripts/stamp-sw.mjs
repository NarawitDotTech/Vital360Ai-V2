// Stamps out/sw.js with a per-build VERSION (hash of every other file in out/), so each deploy gets fresh
// shell caches and the service worker's activate step deletes the old ones. Run after `next build`.
import { createHash } from "node:crypto";
import { readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join, relative } from "node:path";

const out = process.argv[2] ?? "out";
const swPath = join(out, "sw.js");

function walk(dir) {
  return readdirSync(dir).flatMap((n) => {
    const p = join(dir, n);
    return statSync(p).isDirectory() ? walk(p) : [p];
  });
}

const hash = createHash("sha256");
for (const f of walk(out).sort()) {
  if (f === swPath) continue;
  hash.update(relative(out, f)).update(readFileSync(f));
}
const version = hash.digest("hex").slice(0, 12);
const src = readFileSync(swPath, "utf8");
const next = src.replace(/const VERSION = "[^"]*";/, `const VERSION = "${version}";`);
if (next === src && !src.includes(`"${version}"`)) throw new Error("VERSION line not found in " + swPath);
writeFileSync(swPath, next);
console.log(`stamped ${swPath} VERSION=${version}`);
