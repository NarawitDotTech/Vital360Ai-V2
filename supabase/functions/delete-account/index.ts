// Deno Edge Function: permanently delete the calling user's data and auth account.
// Env (set with `supabase secrets set`): ALLOWED_ORIGIN. SUPABASE_URL and
// SUPABASE_SERVICE_ROLE_KEY are injected by Supabase. The service-role key never reaches the client.
import { createClient } from "npm:@supabase/supabase-js@2.45.4";

const allowedOrigin = Deno.env.get("ALLOWED_ORIGIN") ?? "";

const cors = {
  "Access-Control-Allow-Origin": allowedOrigin,
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "authorization, content-type",
  "Access-Control-Max-Age": "86400",
  "Vary": "Origin",
};

function json(status: number, body: Record<string, unknown>): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...cors, "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
}

Deno.serve(async (req: Request): Promise<Response> => {
  const supabaseUrl = Deno.env.get("SUPABASE_URL");
  const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
  if (!allowedOrigin || !supabaseUrl || !serviceKey) {
    return json(500, { error: "server_misconfigured" });
  }

  const origin = req.headers.get("Origin");
  if (origin !== null && origin !== allowedOrigin) {
    return json(403, { error: "origin_not_allowed" });
  }
  if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: cors });
  if (req.method !== "POST") return json(405, { error: "method_not_allowed" });

  const match = /^Bearer (.+)$/.exec(req.headers.get("Authorization") ?? "");
  if (!match) return json(401, { error: "missing_bearer_token" });

  const admin = createClient(supabaseUrl, serviceKey, {
    auth: { autoRefreshToken: false, persistSession: false },
  });

  const { data, error: authError } = await admin.auth.getUser(match[1]);
  if (authError || !data.user) return json(401, { error: "invalid_token" });
  const uid = data.user.id;

  // Explicit deletes (children first); FK cascades would also cover these.
  const steps: Array<[string, string]> = [
    ["feedback", "user_id"],
    ["results", "user_id"],
    ["consent_log", "user_id"],
    ["profiles", "id"],
  ];
  for (const [table, column] of steps) {
    const { error } = await admin.from(table).delete().eq(column, uid);
    if (error) {
      console.error(`delete ${table} failed`, error.message);
      return json(500, { error: "delete_rows_failed", table });
    }
  }

  const { error: delError } = await admin.auth.admin.deleteUser(uid);
  if (delError) {
    console.error("deleteUser failed", delError.message);
    return json(500, { error: "delete_user_failed" });
  }
  return json(200, { deleted: true });
});
