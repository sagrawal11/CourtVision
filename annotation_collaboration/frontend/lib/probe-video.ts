/**
 * Read duration + frame rate from a local File using MediaInfo (WASM).
 *
 * The browser's <video> element does not expose frame rate, so we parse it
 * straight from the container metadata. FrameRate is the video's exact declared
 * rate (e.g. 29.97), so annotators never have to set — or mis-set — it.
 */
import type { GeneralTrack, MediaInfo, VideoTrack } from "mediainfo.js";

// One WASM instance for the page session, reused across files (batch registration).
let mediaInfoPromise: Promise<MediaInfo> | null = null;

function getMediaInfo(): Promise<MediaInfo> {
  if (!mediaInfoPromise) {
    mediaInfoPromise = import("mediainfo.js").then((m) =>
      m.default({
        format: "object",
        // Served from public/ (copied there by scripts/copy-mediainfo-wasm.mjs).
        locateFile: () => "/MediaInfoModule.wasm",
      }),
    );
  }
  return mediaInfoPromise;
}

export async function probeVideoFile(
  file: File,
): Promise<{ duration: number; fps: number; totalFrames: number }> {
  const mediaInfo = await getMediaInfo();

  const result = await mediaInfo.analyzeData(file.size, (chunkSize, offset) =>
    file
      .slice(offset, offset + chunkSize)
      .arrayBuffer()
      .then((buf) => new Uint8Array(buf)),
  );

  const tracks = result.media?.track ?? [];
  const video = tracks.find((t): t is VideoTrack => t["@type"] === "Video");
  const general = tracks.find((t): t is GeneralTrack => t["@type"] === "General");

  const rawFps = video?.FrameRate;
  const duration = video?.Duration ?? general?.Duration;

  if (!rawFps || !Number.isFinite(rawFps) || rawFps <= 0) {
    throw new Error(
      `Could not detect frame rate for ${file.name}. It may be audio-only or an unsupported format.`,
    );
  }
  if (!duration || !Number.isFinite(duration) || duration <= 0) {
    throw new Error(`Could not detect duration for ${file.name}.`);
  }

  const fps = Math.round(rawFps * 1000) / 1000;
  return {
    duration,
    fps,
    totalFrames: Math.max(1, Math.round(duration * fps)),
  };
}

export function stemFromFilename(name: string): string {
  return name.replace(/\.[^.]+$/, "");
}
