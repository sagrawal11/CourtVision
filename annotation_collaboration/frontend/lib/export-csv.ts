import type { AnnotationEvent } from "./supabase";

/** Same columns as cv/tools/annotate.py CSV output. */
export function buildAnnotationsCsv(
  videoId: string,
  events: AnnotationEvent[],
): string {
  const header = "video_id,frame,event_type,stroke_type,point_end_type,player";
  const rows = [...events]
    .sort((a, b) => a.frame - b.frame || a.event_type.localeCompare(b.event_type))
    .map((e) => {
      const cols = [
        videoId,
        String(e.frame),
        e.event_type,
        e.stroke_type || "",
        e.point_end_type || "",
        e.player || "",
      ];
      return cols.join(",");
    });
  return [header, ...rows].join("\n") + "\n";
}

export function downloadCsv(filename: string, content: string) {
  const blob = new Blob([content], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}
