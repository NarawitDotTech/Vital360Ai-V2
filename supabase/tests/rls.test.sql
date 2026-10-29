-- Run with: supabase test db
begin;
create extension if not exists pgtap with schema extensions;
select plan(13);

-- two users (created as the privileged test owner)
insert into auth.users (id, instance_id, aud, role, email)
values ('aaaaaaaa-0000-0000-0000-00000000000a', '00000000-0000-0000-0000-000000000000', 'authenticated', 'authenticated', 'a@example.test'),
       ('bbbbbbbb-0000-0000-0000-00000000000b', '00000000-0000-0000-0000-000000000000', 'authenticated', 'authenticated', 'b@example.test');

select ok((select bool_and(relrowsecurity) from pg_class where oid in
  ('public.profiles'::regclass, 'public.results'::regclass, 'public.consent_log'::regclass, 'public.feedback'::regclass)),
  'RLS enabled on all tables');

-- user A writes
set local role authenticated;
select set_config('request.jwt.claims', '{"sub":"aaaaaaaa-0000-0000-0000-00000000000a","role":"authenticated"}', true);

insert into public.profiles (id, locale) values ('aaaaaaaa-0000-0000-0000-00000000000a', 'th');
insert into public.results (id, user_id, modality, label, tier, confidence, model_version)
values ('11111111-0000-0000-0000-000000000001', 'aaaaaaaa-0000-0000-0000-00000000000a', 'skin', 'mock', 'green', 0.8, 'mock-0');
insert into public.consent_log (user_id, consent_version, locale) values ('aaaaaaaa-0000-0000-0000-00000000000a', 'v1', 'th');
insert into public.feedback (user_id, result_id, helpful) values ('aaaaaaaa-0000-0000-0000-00000000000a', '11111111-0000-0000-0000-000000000001', true);

select is((select count(*)::int from public.results), 1, 'A sees own result');

-- user B
select set_config('request.jwt.claims', '{"sub":"bbbbbbbb-0000-0000-0000-00000000000b","role":"authenticated"}', true);

select is((select count(*)::int from public.profiles), 0, 'B cannot read A profile');
select is((select count(*)::int from public.results), 0, 'B cannot read A results');
select is((select count(*)::int from public.consent_log), 0, 'B cannot read A consent_log');
select is((select count(*)::int from public.feedback), 0, 'B cannot read A feedback');

select throws_ok($$insert into public.results (user_id, modality, label, tier, confidence, model_version)
  values ('aaaaaaaa-0000-0000-0000-00000000000a', 'skin', 'x', 'green', 0.5, 'm')$$,
  '42501', null, 'B cannot insert a result as A');
select throws_ok($$insert into public.feedback (user_id, result_id, helpful)
  values ('bbbbbbbb-0000-0000-0000-00000000000b', '11111111-0000-0000-0000-000000000001', true)$$,
  '23503', null, 'B cannot attach feedback to A result');

update public.profiles set locale = 'en' where id = 'aaaaaaaa-0000-0000-0000-00000000000a';
delete from public.results where user_id = 'aaaaaaaa-0000-0000-0000-00000000000a';
delete from public.feedback where user_id = 'aaaaaaaa-0000-0000-0000-00000000000a';

-- back to A: nothing was modified or deleted by B
select set_config('request.jwt.claims', '{"sub":"aaaaaaaa-0000-0000-0000-00000000000a","role":"authenticated"}', true);
select is((select locale from public.profiles where id = 'aaaaaaaa-0000-0000-0000-00000000000a'), 'th', 'B update did not change A profile');
select is((select count(*)::int from public.results), 1, 'B delete did not remove A result');
select is((select count(*)::int from public.feedback), 1, 'B delete did not remove A feedback');
select throws_ok($$insert into public.results (user_id, modality, label, tier, confidence, model_version)
  values ('aaaaaaaa-0000-0000-0000-00000000000a', 'skin', 'x', 'green', 1.5, 'm')$$,
  '23514', null, 'confidence range enforced');

-- anonymous has no access
reset role;
set local role anon;
select throws_ok($$select * from public.results$$, '42501', null, 'anon denied');

select * from finish();
rollback;
