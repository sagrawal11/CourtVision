/** Mirrors cv/tools/annotate.py — keep in sync for CSV export compatibility. */

export type EventType = "bounce" | "hit" | "point_start" | "point_end";

export const STROKE_TYPES = [
  "forehand",
  "backhand",
  "serve",
  "overhead",
  "volley",
] as const;

export const POINT_END_TYPES = [
  "winner",
  "unforced_error",
  "net_error",
  "forced_error",
] as const;

export type StrokeType = (typeof STROKE_TYPES)[number];
export type PointEndType = (typeof POINT_END_TYPES)[number];

/** CSS colors (approximate BGR palette from annotate.py) */
export function eventColor(
  eventType: string,
  strokeType?: string | null,
  pointEndType?: string | null,
): string {
  if (eventType === "bounce") return "#3296dc";
  if (eventType === "point_start") return "#64ff96";
  if (eventType === "hit") {
    const m: Record<string, string> = {
      forehand: "#32c832",
      backhand: "#dc3232",
      serve: "#dcdc32",
      overhead: "#ffa532",
      volley: "#32dcdc",
    };
    return m[strokeType || "forehand"] || "#c8c8c8";
  }
  if (eventType === "point_end") {
    const m: Record<string, string> = {
      winner: "#ffd732",
      unforced_error: "#ff5032",
      net_error: "#c850b4",
      forced_error: "#ff9664",
    };
    return m[pointEndType || "unforced_error"] || "#c8c8c8";
  }
  return "#c8c8c8";
}

export function eventLabel(
  eventType: string,
  strokeType?: string | null,
  pointEndType?: string | null,
): string {
  if (eventType === "hit") return `HIT: ${strokeType || "?"}`;
  if (eventType === "bounce") return "BOUNCE";
  if (eventType === "point_start") return "POINT START";
  if (eventType === "point_end")
    return `POINT END: ${(pointEndType || "?").replace(/_/g, " ")}`;
  return eventType.toUpperCase();
}

export const CHEAT_SHEET = `
Navigation
  Space / W     play / pause
  A / ←         back 1 frame
  D / →         forward 1 frame
  , / .         back / forward 5 frames
  [ / ]         back / forward 30 frames (~1 s)
  Click timeline to seek

Annotation (at current frame)
  B             Bounce
  H             Hit → then F/K/S/O/V
  P             Point start (serve contact)
  E             Point end → then W/U/N/F
  X             Delete all events at this frame
  Z             Undo last annotation

Hit types (after H)
  F = Forehand   K = bacKhand   S = Serve   O = Overhead   V = Volley

Point end (after E)
  W = Winner   U = Unforced error   N = Net error   F = Forced error

Tips
  • Mark point start on serve contact; hit on racket contact; bounce when ball touches court.
  • Pick a different video from the list than your partner (split by video).
  • Progress saves automatically — close the tab anytime.
`.trim();
