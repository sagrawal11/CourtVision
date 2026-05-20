import { createClient } from "@supabase/supabase-js";

const url = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

if (!url || !key) {
  console.warn(
    "Missing NEXT_PUBLIC_SUPABASE_URL or NEXT_PUBLIC_SUPABASE_ANON_KEY",
  );
}

export const supabase = createClient(url || "", key || "");

/** Legacy cloud uploads only; local-disk mode leaves this null. */
export const STORAGE_BUCKET = "annotation-videos";

export type AnnotationVideo = {
  id: string;
  title: string;
  video_id: string;
  storage_path: string | null;
  expected_filename: string | null;
  fps: number;
  total_frames: number;
  duration_sec: number | null;
  status: "ready" | "in_progress" | "done";
  claimed_by: string | null;
  last_frame: number;
  created_at: string;
  updated_at: string;
};

export type AnnotationEvent = {
  id: string;
  video_uuid: string;
  frame: number;
  event_type: string;
  stroke_type: string | null;
  point_end_type: string | null;
  player: string;
  created_at: string;
};

export function publicVideoUrl(storagePath: string): string {
  return `${url}/storage/v1/object/public/${STORAGE_BUCKET}/${storagePath}`;
}

export function isCloudVideo(video: AnnotationVideo): boolean {
  return Boolean(video.storage_path?.trim());
}
