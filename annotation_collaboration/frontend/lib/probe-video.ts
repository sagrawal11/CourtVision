/** Read duration from a local File via a temporary <video> element. */

export async function probeVideoFile(
  file: File,
  fps: number,
  timeoutMs = 60_000,
): Promise<{ duration: number; totalFrames: number }> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const v = document.createElement("video");
    v.preload = "metadata";

    const timer = window.setTimeout(() => {
      URL.revokeObjectURL(url);
      reject(
        new Error(
          `Timed out reading metadata for ${file.name} (${(file.size / 1e9).toFixed(2)} GB). ` +
            `Register manually with video_id only, or use a smaller proxy file.`,
        ),
      );
    }, timeoutMs);

    v.onloadedmetadata = () => {
      window.clearTimeout(timer);
      const duration = v.duration;
      URL.revokeObjectURL(url);
      if (!Number.isFinite(duration) || duration <= 0) {
        reject(new Error(`Invalid duration for ${file.name}`));
        return;
      }
      resolve({
        duration,
        totalFrames: Math.max(1, Math.round(duration * fps)),
      });
    };
    v.onerror = () => {
      window.clearTimeout(timer);
      URL.revokeObjectURL(url);
      reject(
        new Error(
          `Could not read metadata from ${file.name}. Try manual registration (video_id only) below.`,
        ),
      );
    };
    v.src = url;
  });
}

export function stemFromFilename(name: string): string {
  return name.replace(/\.[^.]+$/, "");
}
