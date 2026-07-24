"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { CheatSheet } from "./CheatSheet";
import {
  POINT_END_TYPES,
  STROKE_TYPES,
  eventColor,
  eventLabel,
} from "@/lib/constants";
import { buildAnnotationsCsv, downloadCsv } from "@/lib/export-csv";
import { probeVideoFile } from "@/lib/probe-video";
import {
  type AnnotationEvent,
  type AnnotationVideo,
  isCloudVideo,
  publicVideoUrl,
  supabase,
} from "@/lib/supabase";

type Mode = "normal" | "await_stroke" | "await_end";

type LocalEvent = {
  id?: string;
  frame: number;
  event_type: string;
  stroke_type: string | null;
  point_end_type: string | null;
  player: string;
};

export function Annotator({ videoId }: { videoId: string }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const saveFrameTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [video, setVideo] = useState<AnnotationVideo | null>(null);
  const [events, setEvents] = useState<LocalEvent[]>([]);
  const [frameIdx, setFrameIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [mode, setMode] = useState<Mode>("normal");
  const [pendingFrame, setPendingFrame] = useState(0);
  const [cheatOpen, setCheatOpen] = useState(false);
  const [fpsEdit, setFpsEdit] = useState(30);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [undoStack, setUndoStack] = useState<(LocalEvent | "delete")[]>([]);
  const [localSrc, setLocalSrc] = useState<string | null>(null);
  const [localFileName, setLocalFileName] = useState<string | null>(null);
  const localSrcRef = useRef<string | null>(null);
  const [admin, setAdmin] = useState(false);

  useEffect(() => {
    setAdmin(new URLSearchParams(window.location.search).get("admin") === "1");
  }, []);

  const totalFrames = video?.total_frames ?? 1;
  const fps = video?.fps ?? 30;
  const usesCloud = video ? isCloudVideo(video) : false;
  const playbackSrc =
    usesCloud && video?.storage_path
      ? publicVideoUrl(video.storage_path)
      : localSrc;

  const load = useCallback(async () => {
    const { data: v, error } = await supabase
      .from("annotation_videos")
      .select("*")
      .eq("id", videoId)
      .single();

    if (error || !v) {
      setLoadError(error?.message || "Video not found");
      return;
    }

    setVideo(v as AnnotationVideo);
    setFpsEdit(v.fps);
    setFrameIdx(v.last_frame || 0);

    if (v.status === "ready") {
      const name =
        typeof window !== "undefined"
          ? localStorage.getItem("annotator_name") || "anonymous"
          : "anonymous";
      await supabase
        .from("annotation_videos")
        .update({ status: "in_progress", claimed_by: name })
        .eq("id", videoId);
    }

    const { data: evs } = await supabase
      .from("annotation_events")
      .select("*")
      .eq("video_uuid", videoId)
      .order("frame");

    setEvents(
      (evs || []).map((e) => ({
        id: e.id,
        frame: e.frame,
        event_type: e.event_type,
        stroke_type: e.stroke_type,
        point_end_type: e.point_end_type,
        player: e.player || "",
      })),
    );
  }, [videoId]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    return () => {
      if (localSrcRef.current) {
        URL.revokeObjectURL(localSrcRef.current);
        localSrcRef.current = null;
      }
    };
  }, []);

  const attachLocalFile = async (file: File) => {
    if (localSrcRef.current) {
      URL.revokeObjectURL(localSrcRef.current);
    }
    const url = URL.createObjectURL(file);
    localSrcRef.current = url;
    setLocalSrc(url);
    setLocalFileName(file.name);

    if (video && !usesCloud) {
      try {
        const { duration, totalFrames } = await probeVideoFile(file, fpsEdit);
        if (totalFrames !== video.total_frames || fpsEdit !== video.fps) {
          await supabase
            .from("annotation_videos")
            .update({
              fps: fpsEdit,
              total_frames: totalFrames,
              duration_sec: duration,
              expected_filename: file.name,
            })
            .eq("id", video.id);
          setVideo({
            ...video,
            fps: fpsEdit,
            total_frames: totalFrames,
            duration_sec: duration,
            expected_filename: file.name,
          });
        }
      } catch (e) {
        console.warn("Could not probe local file:", e);
      }
    }

    if (typeof window !== "undefined" && video) {
      localStorage.setItem(`annotator_local_file_${video.id}`, file.name);
    }
  };

  const seekToFrame = useCallback(
    (idx: number) => {
      const clamped = Math.max(0, Math.min(totalFrames - 1, idx));
      setFrameIdx(clamped);
      const el = videoRef.current;
      if (el && Number.isFinite(el.duration)) {
        el.currentTime = clamped / fps;
      }
    },
    [fps, totalFrames],
  );

  useEffect(() => {
    if (playbackSrc) seekToFrame(frameIdx);
  }, [frameIdx, seekToFrame, video, playbackSrc]);

  const persistLastFrame = useCallback(
    (idx: number) => {
      if (!video) return;
      if (saveFrameTimer.current) clearTimeout(saveFrameTimer.current);
      saveFrameTimer.current = setTimeout(async () => {
        await supabase
          .from("annotation_videos")
          .update({ last_frame: idx })
          .eq("id", video.id);
      }, 800);
    },
    [video],
  );

  useEffect(() => {
    persistLastFrame(frameIdx);
  }, [frameIdx, persistLastFrame]);

  const saveEvent = async (ev: LocalEvent) => {
    if (!video) return;

    await supabase
      .from("annotation_events")
      .delete()
      .eq("video_uuid", video.id)
      .eq("frame", ev.frame)
      .eq("event_type", ev.event_type);

    const { data, error } = await supabase
      .from("annotation_events")
      .insert({
        video_uuid: video.id,
        frame: ev.frame,
        event_type: ev.event_type,
        stroke_type: ev.stroke_type,
        point_end_type: ev.point_end_type,
        player: ev.player || "",
      })
      .select()
      .single();

    if (error) {
      console.error(error);
      return;
    }

    const row = data as AnnotationEvent;
    setEvents((prev) => {
      const filtered = prev.filter(
        (a) => !(a.frame === ev.frame && a.event_type === ev.event_type),
      );
      return [
        ...filtered,
        {
          id: row.id,
          frame: ev.frame,
          event_type: ev.event_type,
          stroke_type: ev.stroke_type,
          point_end_type: ev.point_end_type,
          player: ev.player,
        },
      ].sort((a, b) => a.frame - b.frame);
    });
  };

  const addEvent = async (ev: LocalEvent) => {
    setUndoStack((s) => [...s, ev]);
    await saveEvent(ev);
  };

  const deleteAtFrame = async (frame: number) => {
    if (!video) return;
    setUndoStack((s) => [...s, "delete"]);
    await supabase
      .from("annotation_events")
      .delete()
      .eq("video_uuid", video.id)
      .eq("frame", frame);
    setEvents((prev) => prev.filter((a) => a.frame !== frame));
  };

  const undo = async () => {
    if (!video || undoStack.length === 0) return;
    const last = undoStack[undoStack.length - 1];
    setUndoStack((s) => s.slice(0, -1));

    if (last === "delete") {
      await load();
      return;
    }

    if (last?.id) {
      await supabase.from("annotation_events").delete().eq("id", last.id);
    } else if (last) {
      await supabase
        .from("annotation_events")
        .delete()
        .eq("video_uuid", video.id)
        .eq("frame", last.frame)
        .eq("event_type", last.event_type);
    }
    setEvents((prev) =>
      prev.filter(
        (a) =>
          !(
            a.frame === last.frame &&
            a.event_type === last.event_type
          ),
      ),
    );
  };

  const updateFps = async () => {
    if (!video) return;
    const duration = video.duration_sec || video.total_frames / video.fps;
    const total = Math.max(1, Math.round(duration * fpsEdit));
    await supabase
      .from("annotation_videos")
      .update({ fps: fpsEdit, total_frames: total })
      .eq("id", video.id);
    setVideo({ ...video, fps: fpsEdit, total_frames: total });
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;

      const key = e.key.toLowerCase();

      if (e.key === "Escape") {
        if (mode !== "normal") {
          setMode("normal");
          e.preventDefault();
        }
        return;
      }

      if (mode === "await_stroke") {
        const map: Record<string, string> = {
          f: "forehand",
          k: "backhand",
          s: "serve",
          o: "overhead",
          v: "volley",
        };
        if (map[key]) {
          e.preventDefault();
          addEvent({
            frame: pendingFrame,
            event_type: "hit",
            stroke_type: map[key],
            point_end_type: null,
            player: "",
          });
          setMode("normal");
        }
        return;
      }

      if (mode === "await_end") {
        const map: Record<string, string> = {
          w: "winner",
          u: "unforced_error",
          n: "net_error",
          f: "forced_error",
        };
        if (map[key]) {
          e.preventDefault();
          addEvent({
            frame: pendingFrame,
            event_type: "point_end",
            stroke_type: null,
            point_end_type: map[key],
            player: "",
          });
          setMode("normal");
        }
        return;
      }

      if (key === " ") {
        e.preventDefault();
        setPlaying((p) => !p);
      } else if (key === "a" || e.key === "ArrowLeft") {
        e.preventDefault();
        setPlaying(false);
        setFrameIdx((f) => Math.max(0, f - 1));
      } else if (key === "d" || e.key === "ArrowRight") {
        e.preventDefault();
        setPlaying(false);
        setFrameIdx((f) => Math.min(totalFrames - 1, f + 1));
      } else if (key === ",") {
        e.preventDefault();
        setPlaying(false);
        setFrameIdx((f) => Math.max(0, f - 5));
      } else if (key === ".") {
        e.preventDefault();
        setPlaying(false);
        setFrameIdx((f) => Math.min(totalFrames - 1, f + 5));
      } else if (key === "[") {
        e.preventDefault();
        setPlaying(false);
        setFrameIdx((f) => Math.max(0, f - 30));
      } else if (key === "]") {
        e.preventDefault();
        setPlaying(false);
        setFrameIdx((f) => Math.min(totalFrames - 1, f + 30));
      } else if (key === "b") {
        e.preventDefault();
        addEvent({
          frame: frameIdx,
          event_type: "bounce",
          stroke_type: null,
          point_end_type: null,
          player: "",
        });
      } else if (key === "h") {
        e.preventDefault();
        setPendingFrame(frameIdx);
        setMode("await_stroke");
      } else if (key === "p") {
        e.preventDefault();
        addEvent({
          frame: frameIdx,
          event_type: "point_start",
          stroke_type: null,
          point_end_type: null,
          player: "",
        });
      } else if (key === "e") {
        e.preventDefault();
        setPendingFrame(frameIdx);
        setMode("await_end");
      } else if (key === "x") {
        e.preventDefault();
        deleteAtFrame(frameIdx);
      } else if (key === "z") {
        e.preventDefault();
        undo();
      } else if (key === "w" && !e.shiftKey) {
        // only when not in await_end - handled above
      }
    };

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [mode, frameIdx, pendingFrame, totalFrames, undoStack, video]);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      setFrameIdx((f) => {
        if (f >= totalFrames - 1) {
          setPlaying(false);
          return f;
        }
        return f + 1;
      });
    }, 1000 / fps);
    return () => clearInterval(id);
  }, [playing, fps, totalFrames]);

  const onTimeUpdate = () => {
    const el = videoRef.current;
    if (!el || playing) return;
    const f = Math.round(el.currentTime * fps);
    if (f !== frameIdx) setFrameIdx(f);
  };

  const currentEvents = events.filter((a) => a.frame === frameIdx);

  const stats = {
    total: events.length,
    bounce: events.filter((e) => e.event_type === "bounce").length,
    hit: events.filter((e) => e.event_type === "hit").length,
    point_start: events.filter((e) => e.event_type === "point_start").length,
    point_end: events.filter((e) => e.event_type === "point_end").length,
    frames: new Set(events.map((e) => e.frame)).size,
  };

  if (loadError) {
    return (
      <div style={{ padding: 24 }}>
        <p style={{ color: "#f66" }}>{loadError}</p>
        <Link href="/">← Back</Link>
      </div>
    );
  }

  if (!video) {
    return <p style={{ padding: 24, color: "#888" }}>Loading…</p>;
  }

  const ts = frameIdx / fps;
  const minutes = Math.floor(ts / 60);
  const seconds = ts % 60;

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      <header style={header}>
        <Link href="/" style={{ color: "#50c878", textDecoration: "none" }}>
          ← Videos
        </Link>
        <span style={{ flex: 1, marginLeft: 16, fontWeight: 600 }}>{video.title}</span>
        <button type="button" onClick={() => setCheatOpen(true)} style={hdrBtn}>
          ?
        </button>
        {admin && (
          <button
            type="button"
            onClick={() => {
              const csv = buildAnnotationsCsv(
                video.video_id,
                events.map((e) => ({
                  ...e,
                  id: e.id || "",
                  video_uuid: video.id,
                  created_at: "",
                })) as AnnotationEvent[],
              );
              downloadCsv(`${video.video_id}_annotations.csv`, csv);
            }}
            style={hdrBtn}
          >
            Export CSV
          </button>
        )}
      </header>

      <div style={mainRow}>
        <div style={playerCol}>
          <div style={videoWrap}>
            {!playbackSrc && (
              <div style={openLocalPanel}>
                <p style={{ margin: "0 0 12px", color: "#ccc" }}>
                  Open the MP4 from your disk for <strong>{video.video_id}</strong>
                  {video.expected_filename && (
                    <>
                      <br />
                      <span style={{ fontSize: 13, color: "#888" }}>
                        Expected file: {video.expected_filename}
                      </span>
                    </>
                  )}
                </p>
                {typeof window !== "undefined" &&
                  localStorage.getItem(`annotator_local_file_${video.id}`) && (
                    <p style={{ fontSize: 12, color: "#666", marginBottom: 12 }}>
                      Last session: {localStorage.getItem(`annotator_local_file_${video.id}`)}
                    </p>
                  )}
                <label style={openLocalBtn}>
                  <input
                    type="file"
                    accept="video/*,.mp4,.mov,.webm"
                    style={{ display: "none" }}
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) attachLocalFile(f);
                      e.target.value = "";
                    }}
                  />
                  Choose local video file
                </label>
              </div>
            )}
            {playbackSrc && (
              <video
                ref={videoRef}
                src={playbackSrc}
                style={{ width: "100%", display: "block", background: "#000" }}
                onTimeUpdate={onTimeUpdate}
                onLoadedMetadata={() => seekToFrame(frameIdx)}
                playsInline
              />
            )}
            {localFileName && !usesCloud && (
              <div style={localFileBadge}>
                Local: {localFileName}
                <label style={{ marginLeft: 12, cursor: "pointer", textDecoration: "underline" }}>
                  Change file
                  <input
                    type="file"
                    accept="video/*,.mp4,.mov,.webm"
                    style={{ display: "none" }}
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) attachLocalFile(f);
                      e.target.value = "";
                    }}
                  />
                </label>
              </div>
            )}
            {currentEvents.map((ann, i) => (
              <div
                key={`${ann.frame}-${ann.event_type}-${i}`}
                style={{
                  position: "absolute",
                  left: 0,
                  right: 0,
                  top: i * 28,
                  height: 28,
                  background: eventColor(
                    ann.event_type,
                    ann.stroke_type,
                    ann.point_end_type,
                  ),
                  color: "#000",
                  fontWeight: 600,
                  fontSize: 14,
                  display: "flex",
                  alignItems: "center",
                  paddingLeft: 8,
                  pointerEvents: "none",
                }}
              >
                {eventLabel(ann.event_type, ann.stroke_type, ann.point_end_type)}
              </div>
            ))}
            {mode === "await_stroke" && (
              <PromptBar bg="#283264">
                Hit type: F Forehand · K bacKhand · S Serve · O Overhead · V Volley · Esc cancel
              </PromptBar>
            )}
            {mode === "await_end" && (
              <PromptBar bg="#642828">
                Point end: W Winner · U Unforced · N Net · F Forced · Esc cancel
              </PromptBar>
            )}
          </div>

          <div style={panel}>
            <div style={statusLine}>
              Frame {frameIdx} / {totalFrames - 1} · {minutes}:{seconds.toFixed(2).padStart(5, "0")} ·{" "}
              {playing ? "PLAYING" : mode === "await_stroke" ? "SELECT STROKE" : mode === "await_end" ? "SELECT END" : "PAUSED"} ·{" "}
              {events.length} annotations
            </div>
            <div
              style={scrubber}
              onClick={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                const frac = (e.clientX - rect.left) / rect.width;
                setPlaying(false);
                setFrameIdx(Math.round(frac * (totalFrames - 1)));
              }}
            >
              {events.map((ann, i) => (
                <span
                  key={`${ann.frame}-${ann.event_type}-${i}`}
                  style={{
                    position: "absolute",
                    left: `${(ann.frame / Math.max(totalFrames - 1, 1)) * 100}%`,
                    top: 0,
                    bottom: 0,
                    width: 2,
                    background: eventColor(
                      ann.event_type,
                      ann.stroke_type,
                      ann.point_end_type,
                    ),
                  }}
                />
              ))}
              <span
                style={{
                  position: "absolute",
                  left: `${(frameIdx / Math.max(totalFrames - 1, 1)) * 100}%`,
                  top: -2,
                  bottom: -2,
                  width: 4,
                  marginLeft: -2,
                  background: "#fff",
                }}
              />
            </div>
            <div style={btnRow}>
              <Btn onClick={() => setPlaying((p) => !p)}>Play</Btn>
              <Btn onClick={() => addEvent({ frame: frameIdx, event_type: "bounce", stroke_type: null, point_end_type: null, player: "" })}>Bounce (B)</Btn>
              <Btn onClick={() => { setPendingFrame(frameIdx); setMode("await_stroke"); }}>Hit (H)</Btn>
              <Btn onClick={() => addEvent({ frame: frameIdx, event_type: "point_start", stroke_type: null, point_end_type: null, player: "" })}>Pt start (P)</Btn>
              <Btn onClick={() => { setPendingFrame(frameIdx); setMode("await_end"); }}>Pt end (E)</Btn>
              <Btn onClick={() => deleteAtFrame(frameIdx)}>Del (X)</Btn>
              <Btn onClick={undo}>Undo (Z)</Btn>
            </div>
          </div>
        </div>

        <aside style={sidebar}>
          <h3 style={{ marginTop: 0 }}>Progress</h3>
          <ul style={statList}>
            <li>Total events: {stats.total}</li>
            <li>Bounces: {stats.bounce}</li>
            <li>Hits: {stats.hit}</li>
            <li>Point starts: {stats.point_start}</li>
            <li>Point ends: {stats.point_end}</li>
            <li>Frames with labels: {stats.frames}</li>
            <li>
              Coverage: ~{Math.round((stats.frames / totalFrames) * 100)}% of frames
            </li>
          </ul>
          <h3>FPS</h3>
          <p style={{ fontSize: "0.85rem", color: "#888" }}>
            Should match the video&apos;s real frame rate so labels land on the right frame.
          </p>
          <input
            type="number"
            value={fpsEdit}
            onChange={(e) => setFpsEdit(Number(e.target.value) || 30)}
            style={fpsInput}
          />
          <button type="button" onClick={updateFps} style={hdrBtn}>
            Update FPS
          </button>
          {mode === "await_stroke" && (
            <div style={{ marginTop: 16 }}>
              <p>Stroke type:</p>
              {STROKE_TYPES.map((s) => (
                <button
                  key={s}
                  type="button"
                  style={{ ...strokeBtn, display: "block", width: "100%", marginBottom: 4 }}
                  onClick={() => {
                    addEvent({
                      frame: pendingFrame,
                      event_type: "hit",
                      stroke_type: s,
                      point_end_type: null,
                      player: "",
                    });
                    setMode("normal");
                  }}
                >
                  {s}
                </button>
              ))}
            </div>
          )}
          {mode === "await_end" && (
            <div style={{ marginTop: 16 }}>
              <p>Point end:</p>
              {POINT_END_TYPES.map((s) => (
                <button
                  key={s}
                  type="button"
                  style={{ ...strokeBtn, display: "block", width: "100%", marginBottom: 4 }}
                  onClick={() => {
                    addEvent({
                      frame: pendingFrame,
                      event_type: "point_end",
                      stroke_type: null,
                      point_end_type: s,
                      player: "",
                    });
                    setMode("normal");
                  }}
                >
                  {s.replace(/_/g, " ")}
                </button>
              ))}
            </div>
          )}
        </aside>
      </div>

      <CheatSheet open={cheatOpen} onClose={() => setCheatOpen(false)} />
    </div>
  );
}

function Btn({
  children,
  onClick,
}: {
  children: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <button type="button" onClick={onClick} style={smallBtn}>
      {children}
    </button>
  );
}

function PromptBar({ bg, children }: { bg: string; children: React.ReactNode }) {
  return (
    <div
      style={{
        position: "absolute",
        bottom: 0,
        left: 0,
        right: 0,
        padding: "8px 12px",
        background: bg,
        color: "#eee",
        fontSize: 13,
      }}
    >
      {children}
    </div>
  );
}

const header: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  padding: "12px 20px",
  borderBottom: "1px solid #333",
  background: "#141414",
};

const hdrBtn: React.CSSProperties = {
  marginLeft: 8,
  padding: "6px 12px",
  background: "#333",
  border: "1px solid #555",
  borderRadius: 6,
  color: "#eee",
};

const mainRow: React.CSSProperties = {
  display: "flex",
  flex: 1,
  gap: 0,
  overflow: "hidden",
};

const playerCol: React.CSSProperties = {
  flex: 1,
  display: "flex",
  flexDirection: "column",
  minWidth: 0,
};

const videoWrap: React.CSSProperties = {
  position: "relative",
  background: "#000",
  flex: 1,
  maxHeight: "calc(100vh - 200px)",
  overflow: "hidden",
};

const panel: React.CSSProperties = {
  background: "#141414",
  borderTop: "1px solid #333",
  padding: 12,
};

const statusLine: React.CSSProperties = {
  fontSize: 13,
  color: "#aaa",
  marginBottom: 8,
};

const scrubber: React.CSSProperties = {
  position: "relative",
  height: 20,
  background: "#333",
  borderRadius: 4,
  marginBottom: 10,
  cursor: "pointer",
};

const btnRow: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: 6,
};

const smallBtn: React.CSSProperties = {
  padding: "6px 10px",
  fontSize: 12,
  background: "#2a2a2a",
  border: "1px solid #444",
  borderRadius: 4,
  color: "#ddd",
};

const sidebar: React.CSSProperties = {
  width: 260,
  padding: 16,
  borderLeft: "1px solid #333",
  background: "#141414",
  overflowY: "auto",
};

const statList: React.CSSProperties = {
  paddingLeft: 18,
  fontSize: 14,
  color: "#bbb",
  lineHeight: 1.6,
};

const fpsInput: React.CSSProperties = {
  width: "100%",
  padding: 8,
  background: "#111",
  border: "1px solid #444",
  borderRadius: 6,
  color: "#fff",
  marginBottom: 8,
};

const strokeBtn: React.CSSProperties = {
  padding: "8px 10px",
  background: "#2a3a2a",
  border: "1px solid #50c878",
  borderRadius: 6,
  color: "#eee",
  textAlign: "left",
};

const openLocalPanel: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  minHeight: 280,
  padding: 24,
  textAlign: "center",
};

const openLocalBtn: React.CSSProperties = {
  padding: "12px 20px",
  background: "#50c878",
  color: "#000",
  borderRadius: 8,
  fontWeight: 600,
  cursor: "pointer",
};

const localFileBadge: React.CSSProperties = {
  position: "absolute",
  top: 8,
  right: 8,
  background: "rgba(0,0,0,0.75)",
  padding: "6px 10px",
  borderRadius: 6,
  fontSize: 12,
  color: "#aaa",
};
