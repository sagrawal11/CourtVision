"use client";

import { useEffect, useState } from "react";
import { checkSupabaseConnection } from "@/lib/check-supabase";

export function SupabaseConnectionBanner() {
  const [result, setResult] = useState<Awaited<
    ReturnType<typeof checkSupabaseConnection>
  > | null>(null);

  useEffect(() => {
    checkSupabaseConnection().then(setResult);
  }, []);

  if (!result) return null;

  return (
    <div
      style={{
        marginBottom: 16,
        padding: "10px 14px",
        borderRadius: 8,
        fontSize: "0.9rem",
        background: result.ok ? "rgba(80,200,120,0.12)" : "rgba(255,80,80,0.12)",
        border: `1px solid ${result.ok ? "#50c878" : "#a44"}`,
        color: result.ok ? "#8fd4a8" : "#f99",
      }}
    >
      {result.ok ? "✓ " : "✗ "}
      {result.message}
      {!result.ok && (
        <div style={{ marginTop: 8, fontSize: "0.8rem", color: "#ccc" }}>
          Fix <code style={{ color: "#50c878" }}>frontend/.env.local</code> using
          Dashboard → Project Settings → API → Project URL + anon public key.
          Restart <code style={{ color: "#50c878" }}>npm run dev</code> after edits.
        </div>
      )}
    </div>
  );
}
