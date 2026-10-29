-- Vitals 360 AI: optional sync backend. Summaries only, never media.
-- RLS is enabled on every table; the app works fully without this backend.

create table public.profiles (
  id uuid primary key references auth.users (id) on delete cascade,
  locale text not null default 'th' check (locale in ('th', 'en')),
  created_at timestamptz not null default now()
);

create table public.results (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  modality text not null check (modality in ('skin', 'audio', 'resp', 'fusion')),
  label text not null check (char_length(label) between 1 and 40),
  tier text not null check (tier in ('green', 'amber', 'red', 'uncertain')),
  confidence real not null check (confidence >= 0 and confidence <= 1),
  breaths_per_min real check (breaths_per_min is null or (breaths_per_min > 0 and breaths_per_min <= 120)),
  model_version text not null check (char_length(model_version) between 1 and 64),
  created_at timestamptz not null default now(),
  unique (id, user_id)
);
create index results_user_id_idx on public.results (user_id, created_at desc);

create table public.consent_log (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  consent_version text not null check (char_length(consent_version) between 1 and 32),
  accepted_at timestamptz not null default now(),
  locale text not null check (locale in ('th', 'en'))
);
create index consent_log_user_id_idx on public.consent_log (user_id);

create table public.feedback (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  result_id uuid not null,
  helpful boolean not null,
  comment text check (comment is null or char_length(comment) <= 500),
  created_at timestamptz not null default now(),
  -- the referenced result must belong to the same user
  foreign key (result_id, user_id) references public.results (id, user_id) on delete cascade
);
create index feedback_user_id_idx on public.feedback (user_id);
create index feedback_result_id_idx on public.feedback (result_id);

alter table public.profiles enable row level security;
alter table public.results enable row level security;
alter table public.consent_log enable row level security;
alter table public.feedback enable row level security;

-- No anonymous access at all.
revoke all on public.profiles, public.results, public.consent_log, public.feedback from anon;

-- profiles: own row only
create policy profiles_select on public.profiles for select to authenticated using (auth.uid() = id);
create policy profiles_insert on public.profiles for insert to authenticated with check (auth.uid() = id);
create policy profiles_update on public.profiles for update to authenticated using (auth.uid() = id) with check (auth.uid() = id);
create policy profiles_delete on public.profiles for delete to authenticated using (auth.uid() = id);

-- results: own rows; immutable once written (no update policy)
create policy results_select on public.results for select to authenticated using (auth.uid() = user_id);
create policy results_insert on public.results for insert to authenticated with check (auth.uid() = user_id);
create policy results_delete on public.results for delete to authenticated using (auth.uid() = user_id);

-- consent_log: append-only audit trail (select + insert). Deleted via account deletion cascade.
create policy consent_log_select on public.consent_log for select to authenticated using (auth.uid() = user_id);
create policy consent_log_insert on public.consent_log for insert to authenticated with check (auth.uid() = user_id);

-- feedback: opt-in, own rows
create policy feedback_select on public.feedback for select to authenticated using (auth.uid() = user_id);
create policy feedback_insert on public.feedback for insert to authenticated with check (auth.uid() = user_id);
create policy feedback_update on public.feedback for update to authenticated using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy feedback_delete on public.feedback for delete to authenticated using (auth.uid() = user_id);
