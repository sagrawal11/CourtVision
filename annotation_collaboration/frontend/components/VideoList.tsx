"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import {
  STORAGE_BUCKET,
  type AnnotationEvent,
  type AnnotationVideo,
  isCloudVideo,
  supabase,
} from "@/lib/supabase";
import { buildAnnotationsCsv, downloadCsv } from "@/lib/export-csv";

type VideoWithStats = AnnotationVideo & {
  eventCount: number;
  framesWithEvents: number;
};

export function VideoList({ admin = false }: { admin?: boolean }) {
  const [videos, setVideos] = useState<VideoWithStats[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<"all" | "ready" | "in_progress" | "done">("all");
  const [name, setName] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    const { data: vids, error } = await supabase
      .from("annotation_videos")
      .select("*")
      .order("created_at", { ascending: false });

    if (error) {
      console.error(error);
      setLoading(false);
      return;
    }

    const list = vids || [];
    const withStats: VideoWithStats[] = [];

    for (const v of list) {
      const { data: evs } = await supabase
        .from("annotation_events")
        .select("frame")
        .eq("video_uuid", v.id);
      const frames = new Set((evs || []).map((e) => e.frame));
      withStats.push({
        ...v,
        eventCount: evs?.length ?? 0,
        framesWithEvents: frames.size,
      });
    }

    setVideos(withStats);
    setLoading(false);
  }, []);

  useEffect(() => {
    load();
    if (typeof window !== "undefined") {
      setName(localStorage.getItem("annotator_name") || "");
    }
  }, [load]);

  const saveName = (n: string) => {
    setName(n);
    localStorage.setItem("annotator_name", n);
  };

  const claim = async (v: AnnotationVideo) => {
    await supabase
      .from("annotation_videos")
      .update({ status: "in_progress", claimed_by: name || "anonymous" })
      .eq("id", v.id);
    load();
  };

  const markDone = async (v: AnnotationVideo) => {
    await supabase.from("annotation_videos").update({ status: "done" }).eq("id", v.id);
    load();
  };

  const exportVideo = async (v: AnnotationVideo) => {
    const { data: evs } = await supabase
      .from("annotation_events")
      .select("*")
      .eq("video_uuid", v.id);
    const csv = buildAnnotationsCsv(v.video_id, (evs || []) as AnnotationEvent[]);
    downloadCsv(`${v.video_id}_annotations.csv`, csv);
  };

  const deleteVideo = async (v: AnnotationVideo) => {
    if (!confirm(`Delete "${v.title}" and all annotations?`)) return;
    await supabase.from("annotation_videos").delete().eq("id", v.id);
    if (isCloudVideo(v) && v.storage_path) {
      await supabase.storage.from(STORAGE_BUCKET).remove([v.storage_path]);
    }
    load();
  };

  const filtered = videos.filter((v) => filter === "all" || v.status === filter);

  return (
    <section>
      <div style={toolbar}>
        <label style={{ fontSize: "0.9rem" }}>
          Your name (for claiming videos)
          <input
            value={name}
            onChange={(e) => saveName(e.target.value)}
            placeholder="e.g. Sarthak"
            style={nameInput}
          />
        </label>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {(["all", "ready", "in_progress", "done"] as const).map((f) => (
            <button
              key={f}
              type="button"
              onClick={() => setFilter(f)}
              style={filter === f ? filterActive : filterBtn}
            >
              {f.replace("_", " ")}
            </button>
          ))}
        </div>
      </div>

      {loading ? (
        <p style={{ color: "#888" }}>Loading…</p>
      ) : filtered.length === 0 ? (
        <p style={{ color: "#888" }}>No videos to annotate yet.</p>
      ) : (
        <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
          {filtered.map((v) => (
            <li key={v.id} style={card}>
              <div style={cardMain}>
                <div>
                  <strong>{v.title}</strong>
                  <div style={meta}>
                    <span style={badge(v.status)}>{v.status}</span>
                    {v.claimed_by && <span> · {v.claimed_by}</span>}
                    <span>
                      {" "}
                      · {v.eventCount} events · {v.framesWithEvents} frames touched
                      {v.total_frames > 0 && (
                        <> · ~{Math.round((v.framesWithEvents / v.total_frames) * 100)}% frame coverage</>
                      )}
                    </span>
                  </div>
                  <div style={{ fontSize: "0.8rem", color: "#666", marginTop: 4 }}>
                    {v.fps} fps · {v.total_frames} frames
                    {v.expected_filename && (
                      <> · open file: {v.expected_filename}</>
                    )}
                  </div>
                </div>
                <div style={actions}>
                  <Link
                    href={`/annotate/${v.id}${admin ? "?admin=1" : ""}`}
                    style={primaryBtn}
                  >
                    {v.status === "done" ? "Review" : "Annotate"}
                  </Link>
                  {v.status === "ready" && (
                    <button type="button" onClick={() => claim(v)} style={secBtn}>
                      Start
                    </button>
                  )}
                  {v.status === "in_progress" && (
                    <button type="button" onClick={() => markDone(v)} style={secBtn}>
                      Mark done
                    </button>
                  )}
                  {admin && (
                    <>
                      <button type="button" onClick={() => exportVideo(v)} style={secBtn}>
                        Export CSV
                      </button>
                      <button type="button" onClick={() => deleteVideo(v)} style={dangerBtn}>
                        Delete
                      </button>
                    </>
                  )}
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function badge(status: string): React.CSSProperties {
  const colors: Record<string, string> = {
    ready: "#444",
    in_progress: "#50c878",
    done: "#2a5a3a",
  };
  return {
    display: "inline-block",
    padding: "2px 8px",
    borderRadius: 4,
    fontSize: "0.75rem",
    background: colors[status] || "#444",
    textTransform: "uppercase",
  };
}

const toolbar: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: 16,
  alignItems: "flex-end",
  marginBottom: 16,
};

const nameInput: React.CSSProperties = {
  display: "block",
  marginTop: 4,
  padding: "8px 10px",
  width: 200,
  background: "#111",
  border: "1px solid #444",
  borderRadius: 6,
  color: "#fff",
};

const filterBtn: React.CSSProperties = {
  padding: "6px 12px",
  background: "#222",
  border: "1px solid #444",
  borderRadius: 6,
  color: "#ccc",
};

const filterActive: React.CSSProperties = {
  ...filterBtn,
  background: "#50c878",
  color: "#000",
  borderColor: "#50c878",
};

const card: React.CSSProperties = {
  background: "#1a1a1a",
  border: "1px solid #333",
  borderRadius: 10,
  marginBottom: 12,
  padding: 16,
};

const cardMain: React.CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  gap: 16,
  flexWrap: "wrap",
};

const meta: React.CSSProperties = { fontSize: "0.85rem", color: "#999", marginTop: 6 };

const actions: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: 8,
  alignItems: "center",
};

const primaryBtn: React.CSSProperties = {
  padding: "8px 16px",
  background: "#50c878",
  color: "#000",
  borderRadius: 6,
  fontWeight: 600,
  textDecoration: "none",
};

const secBtn: React.CSSProperties = {
  padding: "8px 12px",
  background: "#333",
  border: "1px solid #555",
  borderRadius: 6,
  color: "#eee",
};

const dangerBtn: React.CSSProperties = {
  ...secBtn,
  color: "#f88",
  borderColor: "#633",
};
