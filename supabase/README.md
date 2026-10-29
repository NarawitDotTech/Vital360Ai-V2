# Supabase (optional backend)

The app is local-first and works without any of this. Needs the [Supabase CLI](https://supabase.com/docs/guides/cli) and Docker (tests only).

```bash
supabase login
supabase link --project-ref <your-project-ref>
supabase db push                                   # applies supabase/migrations/*
supabase secrets set ALLOWED_ORIGIN=https://your-site.pages.dev
supabase functions deploy delete-account
supabase start && supabase test db                 # local pgTAP RLS tests
```

Then put the project URL and anon key in `.env.local` (see `.env.example`). Never put the service-role key in the app or in git.
