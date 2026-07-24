"use client";

import { useEffect, useState } from "react";
import { CheatSheet } from "@/components/CheatSheet";
import { VideoList } from "@/components/VideoList";
import { RegisterVideos } from "@/components/RegisterVideos";

export default function HomePage() {
  const [cheatOpen, setCheatOpen] = useState(false);
  const [listKey, setListKey] = useState(0);
  const [admin, setAdmin] = useState(false);

  useEffect(() => {
    setAdmin(new URLSearchParams(window.location.search).get("admin") === "1");
  }, []);

  return (
    <main style={{ maxWidth: 960, margin: "0 auto", padding: "24px 20px" }}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
        <div>
          <h1 style={{ margin: 0, fontSize: "1.75rem" }}>Tennis Annotation</h1>
          <p style={{ margin: "8px 0 0", color: "#888", fontSize: "0.95rem" }}>
            Pick a video below, open its file from your computer, and label the
            action frame by frame. Your progress saves automatically.
          </p>
        </div>
        <button type="button" onClick={() => setCheatOpen(true)} style={helpBtn}>
          Cheat sheet ?
        </button>
      </header>

      {admin && <RegisterVideos onRegistered={() => setListKey((k) => k + 1)} />}
      <VideoList key={listKey} admin={admin} />

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
