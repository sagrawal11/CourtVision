"use client";

import { useState } from "react";
import { CheatSheet } from "@/components/CheatSheet";
import { VideoList } from "@/components/VideoList";
import { RegisterVideos } from "@/components/RegisterVideos";
import { SupabaseConnectionBanner } from "@/components/SupabaseConnectionBanner";

export default function HomePage() {
  const [cheatOpen, setCheatOpen] = useState(false);
  const [listKey, setListKey] = useState(0);

  return (
    <main style={{ maxWidth: 960, margin: "0 auto", padding: "24px 20px" }}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
        <div>
          <h1 style={{ margin: 0, fontSize: "1.75rem" }}>Tennis Annotation</h1>
          <p style={{ margin: "8px 0 0", color: "#888", fontSize: "0.95rem" }}>
            Videos on your disk · annotations sync to Supabase · export CSV for{" "}
            <code style={{ color: "#50c878" }}>cv/training_data</code>
          </p>
        </div>
        <button type="button" onClick={() => setCheatOpen(true)} style={helpBtn}>
          Cheat sheet ?
        </button>
      </header>

      <SupabaseConnectionBanner />
      <RegisterVideos onRegistered={() => setListKey((k) => k + 1)} />
      <VideoList key={listKey} />

      <CheatSheet open={cheatOpen} onClose={() => setCheatOpen(false)} />
    </main>
  );
}

const helpBtn: React.CSSProperties = {
  padding: "10px 16px",
  background: "#333",
  border: "1px solid #555",
  borderRadius: 8,
  color: "#eee",
};
