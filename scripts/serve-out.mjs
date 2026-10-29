// Tiny dependency-free static server for the exported site in out/. Usage: node scripts/serve-out.mjs [port]
import { createServer } from "node:http";
import { createReadStream, existsSync, statSync } from "node:fs";
import { extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(fileURLToPath(new URL("../out", import.meta.url)));
const port = Number(process.argv[2] ?? process.env.PORT ?? 4173);

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
  ".webmanifest": "application/manifest+json",
  ".woff2": "font/woff2",
  ".wasm": "application/wasm",
  ".onnx": "application/octet-stream",
};

function resolveFile(pathname) {
  const p = normalize(decodeURIComponent(pathname)).replace(/^(\.\.[/\\])+/, "");
  const abs = join(root, p);
  if (abs !== root && !abs.startsWith(root + sep)) return null;
  if (existsSync(abs) && statSync(abs).isDirectory()) {
    const idx = join(abs, "index.html");
    return existsSync(idx) ? idx : null;
  }
  if (existsSync(abs)) return abs;
  if (existsSync(abs + ".html")) return abs + ".html";
  return null;
}

createServer((req, res) => {
  let pathname;
  try {
    pathname = new URL(req.url ?? "/", "http://localhost").pathname;
  } catch {
    res.writeHead(400).end("bad request");
    return;
  }
  const file = resolveFile(pathname);
  if (!file) {
    const nf = join(root, "404.html");
    res.writeHead(404, { "content-type": TYPES[".html"] });
    if (existsSync(nf)) createReadStream(nf).pipe(res);
    else res.end("Not found");
    return;
  }
  const headers = { "content-type": TYPES[extname(file)] ?? "application/octet-stream" };
  if (file.includes(`${sep}_next${sep}static${sep}`)) headers["cache-control"] = "public, max-age=31536000, immutable";
  res.writeHead(200, headers);
  if (req.method === "HEAD") return res.end();
  createReadStream(file).pipe(res);
}).listen(port, () => console.log(`serving ${root} on http://localhost:${port}`));
