# Vitals 360 AI

Free, privacy-first health **screening** PWA (Thai first, English second) by Synapse Logic. Skin photo, cough/breath audio and chest-video breathing rate are analysed **on the device**; nothing is uploaded. Screening only, not a diagnosis.

Status: app complete with **demo (mock) models**, clearly labelled in the UI. Breathing rate uses real signal processing. Real trained models drop in via `public/models/manifest.json` (see `docs/agents/inference.md`).

## Run
```bash
npm ci
npm run dev            # http://localhost:3000/th/
npm run build          # static site in out/
npm run lint && npm run typecheck && npm test
npx playwright test    # smoke test against out/
npm run icons          # regenerate icons from assets/icon.svg
```

Deploy: upload `out/` to Cloudflare Pages (headers in `public/_headers`). Optional Supabase backend: `supabase/README.md`.

Docs: `docs/plan.md` (scope and decisions), `docs/agents/*.md` (per-area notes), `docs/brief/` (original brief).
