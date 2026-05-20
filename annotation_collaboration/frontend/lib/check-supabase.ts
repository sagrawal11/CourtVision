import { supabase } from "./supabase";

export type ConnectionCheck =
  | { ok: true; message: string }
  | { ok: false; message: string };

export async function checkSupabaseConnection(): Promise<ConnectionCheck> {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL || "";
  const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || "";

  if (!url || url.includes("YOUR_PROJECT")) {
    return {
      ok: false,
      message: "Missing NEXT_PUBLIC_SUPABASE_URL in .env.local",
    };
  }
  if (!key || key.includes("your_anon")) {
    return {
      ok: false,
      message: "Missing NEXT_PUBLIC_SUPABASE_ANON_KEY in .env.local",
    };
  }
  if (!url.startsWith("https://") || !url.includes(".supabase.co")) {
    return {
      ok: false,
      message:
        "NEXT_PUBLIC_SUPABASE_URL should look like https://YOUR_REF.supabase.co (no trailing slash)",
    };
  }

  try {
    const refFromUrl = url.match(/https:\/\/([^.]+)\.supabase\.co/)?.[1];
    const payload = key.split(".")[1];
    if (payload && refFromUrl) {
      const json = JSON.parse(atob(payload.replace(/-/g, "+").replace(/_/g, "/")));
      if (json.ref && json.ref !== refFromUrl) {
        return {
          ok: false,
          message: `URL project (${refFromUrl}) does not match anon key project (${json.ref}). Copy both from the same Supabase project.`,
        };
      }
    }
  } catch {
    /* non-fatal parse issue */
  }

  const { error } = await supabase.from("annotation_videos").select("id").limit(1);
  if (error) {
    if (
      error.message.includes("relation") ||
      error.message.includes("annotation_videos")
    ) {
      return {
        ok: false,
        message: `Connected, but tables missing — run supabase/schema.sql (and migration_local_disk.sql if needed). (${error.message})`,
      };
    }
    return { ok: false, message: error.message };
  }

  return { ok: true, message: `Connected to ${url.replace("https://", "")}` };
}
