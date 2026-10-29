# Backend agent notes

## Decisions
- Schema: `profiles`, `results` (summaries only, enum/range checks), `consent_log` (append-only: select+insert), `feedback` (opt-in, comment <= 500 chars). All FKs cascade from `auth.users`. `feedback` uses a composite FK `(result_id, user_id)` so feedback can only reference the same user's result. `results` has no update policy (immutable). `anon` has no grants.
- Modality enum: skin, audio, resp, fusion. Tier enum: green, amber, red, uncertain. Label free text <= 40 chars. Aligned with src/inference/types.ts.
- delete-account: validates the JWT via `auth.admin.getUser`, deletes rows child-first, then the auth user. Service-role key comes from Edge Function env only. Single allowed origin (`ALLOWED_ORIGIN`), POST only. No rate limit beyond Supabase's own gateway limits (gap).
- CSP: `script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval'`. Next's static export emits inline scripts (`self.__next_f.push(...)` flight data, ~7.5 KB on the index page) whose content differs per page and per build. Hashes would need per-page header rules regenerated after every build (`_headers` is static, Cloudflare has no nonce support for static assets). Trade-off: `unsafe-inline` weakens XSS protection for scripts; mitigated by no user-HTML rendering, `object-src 'none'`, `base-uri 'self'`, `frame-ancestors 'none'`, and strict connect-src. Upgrade path: a post-build script that hashes inline scripts per route and generates `_headers`. `style-src 'unsafe-inline'` is kept for React inline style attributes.
- Replace `YOUR-PROJECT-REF.supabase.co` in `public/_headers` connect-src with the real URL before deploy.
- CI: Lighthouse via `@lhci/cli@0.15.1` with `staticDistDir: ./out`. Script size budget is a `warn` (transfer size of the lhci static server, which may not gzip, so treat as approximate). CI runs on PRs and non-main pushes only.

## Verified
- Migration applied cleanly on a real local PostgreSQL 16 (with stubbed `auth` schema and roles).
- RLS tests: pgTAP is not installed here, so I ran the test file against a tiny shim of plan/ok/is/throws_ok; all 13 assertions passed (re-run after enum alignment). Real `supabase test db` not run (no CLI/Docker).
- `lighthouserc.json` parses; `ci.yml` parses (python yaml).
- Inspected existing `out/index.html` for inline scripts.

## Not verified
- Edge Function not type-checked or run (no deno). supabase-js version pin `2.45.4` not checked against the registry.
- Action tags `actions/checkout@v4`, `actions/setup-node@v4` chosen from knowledge; GitHub API was not reachable to confirm (newer majors may exist).
- CSP not tested in a browser against the final build; `public/` was empty when I worked (no sw.js/manifest yet). Lighthouse thresholds not run.
