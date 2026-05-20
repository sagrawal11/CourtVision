"use client";

import { useState } from "react";
import { formatError, isSchemaMigrationError } from "@/lib/format-error";
import { probeVideoFile, stemFromFilename } from "@/lib/probe-video";
import { supabase } from "@/lib/supabase";

export function RegisterVideos({ onRegistered }: { onRegistered: () => void }) {
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [fps, setFps] = useState(30);
  const [error, setError] = useState<string | null>(null);

  const [manualId, setManualId] = useState("");
  const [manualTitle, setManualTitle] = useState("");

  const registerRow = async (opts: {
    video_id: string;
    title: string;
    expected_filename: string | null;
    fps: number;
    total_frames: number;
    duration_sec: number | null;
  }) => {
    const { error: dbErr } = await supabase.from("annotation_videos").insert({
      title: opts.title,
      video_id: opts.video_id,
      storage_path: null,
      expected_filename: opts.expected_filename,
      fps: opts.fps,
      total_frames: opts.total_frames,
      duration_sec: opts.duration_sec,
      status: "ready",
      last_frame: 0,
    });
    if (dbErr) {
      if (dbErr.code === "23505") {
        throw new Error(`video_id already registered: ${opts.video_id}`);
      }
      const msg = formatError(dbErr);
      if (isSchemaMigrationError(msg)) {
        throw new Error(
          `${msg} — Run annotation_collaboration/supabase/migration_local_disk.sql in the Supabase SQL Editor.`,
        );
      }
      throw new Error(msg);
    }
  };

  const handleFiles = async (files: FileList | null) => {
    if (!files?.length) return;
    setError(null);
    setBusy(true);

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      if (!file.type.startsWith("video/") && !file.name.match(/\.(mp4|mov|webm|avi)$/i)) {
        continue;
      }

      setStatus(`Registering ${i + 1}/${files.length}: ${file.name}…`);

      try {
        const videoIdStem = stemFromFilename(file.name);
        const { duration, totalFrames } = await probeVideoFile(file, fps);
        await registerRow({
          video_id: videoIdStem,
          title: file.name,
          expected_filename: file.name,
          fps,
          total_frames: totalFrames,
          duration_sec: duration,
        });
      } catch (e) {
        setError(formatError(e));
        setBusy(false);
        setStatus("");
        return;
      }
    }

    setStatus("");
    setBusy(false);
    onRegistered();
  };

  const handleManual = async (e: React.FormEvent) => {
    e.preventDefault();
    const id = manualId.trim();
    if (!id) {
      setError("video_id is required (must match your MP4 filename stem)");
      return;
    }
    setError(null);
    setBusy(true);
    setStatus("Registering…");

    try {
      await registerRow({
        video_id: id,
        title: manualTitle.trim() || `${id}.mp4`,
        expected_filename: manualTitle.trim() ? null : `${id}.mp4`,
        fps,
        total_frames: 1,
        duration_sec: null,
      });
      setManualId("");
      setManualTitle("");
      onRegistered();
    } catch (err) {
      setError(formatError(err));
    } finally {
      setBusy(false);
      setStatus("");
    }
  };

  return (
    <section style={box}>
      <h2 style={h2}>Register videos (local disk)</h2>
      <p style={hint}>
        Videos stay on your computer — only labels sync to Supabase. Share MP4s with your
        collaborator separately (Drive, AirDrop, etc.). Pick one video per person on the list
        below so you do not overlap.
      </p>

      <label style={label}>
        FPS for frame indexing (same for all registrations)
        <input
          type="number"
          min={1}
          max={120}
          step={0.01}
          value={fps}
          disabled={busy}
          onChange={(e) => setFps(Number(e.target.value) || 30)}
          style={input}
        />
      </label>

      <label style={dropZone}>
        <input
          type="file"
          accept="video/*,.mp4,.mov,.webm"
          multiple
          disabled={busy}
          style={{ display: "none" }}
          onChange={(e) => {
            handleFiles(e.target.files);
            e.target.value = "";
          }}
        />
        <span>
          {busy && status
            ? status
            : "Register from local MP4s (reads metadata only — no upload)"}
        </span>
      </label>

      <form onSubmit={handleManual} style={{ marginTop: 16 }}>
        <p style={{ ...hint, marginBottom: 8 }}>Or register by name only (probe FPS later in annotator):</p>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <input
            placeholder="video_id e.g. Indoor Match 1 15.53.25"
            value={manualId}
            onChange={(e) => setManualId(e.target.value)}
            disabled={busy}
            style={{ ...input, flex: "1 1 200px", width: "auto" }}
          />
          <input
            placeholder="Title (optional)"
            value={manualTitle}
            onChange={(e) => setManualTitle(e.target.value)}
            disabled={busy}
            style={{ ...input, flex: "1 1 160px", width: "auto" }}
          />
          <button type="submit" disabled={busy} style={submitBtn}>
            Add
          </button>
        </div>
      </form>

      {error && <p style={{ color: "#f66", marginTop: 8 }}>{error}</p>}
    </section>
  );
}

const box: React.CSSProperties = {
  background: "#1a1a1a",
  border: "1px solid #333",
  borderRadius: 12,
  padding: 20,
  marginBottom: 24,
};

const h2: React.CSSProperties = { margin: "0 0 8px", fontSize: "1.1rem" };
const hint: React.CSSProperties = { margin: "0 0 12px", color: "#888", fontSize: "0.9rem" };
const label: React.CSSProperties = { display: "block", marginBottom: 12, fontSize: "0.9rem" };
const input: React.CSSProperties = {
  display: "block",
  marginTop: 4,
  padding: "8px 10px",
  width: 120,
  background: "#111",
  border: "1px solid #444",
  borderRadius: 6,
  color: "#fff",
};
const dropZone: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  minHeight: 88,
  border: "2px dashed #444",
  borderRadius: 8,
  cursor: "pointer",
  color: "#aaa",
  textAlign: "center",
  padding: 16,
};
const submitBtn: React.CSSProperties = {
  padding: "8px 16px",
  background: "#50c878",
  color: "#000",
  border: "none",
  borderRadius: 6,
  fontWeight: 600,
};
