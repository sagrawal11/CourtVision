// Copies the MediaInfo WASM binary into public/ so it can be served at
// /MediaInfoModule.wasm. Runs on postinstall (locally and on Vercel) so the
// binary is always fresh from the installed package and never committed to git.
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = resolve(here, "../node_modules/mediainfo.js/dist/MediaInfoModule.wasm");
const destDir = resolve(here, "../public");
const dest = resolve(destDir, "MediaInfoModule.wasm");

if (!existsSync(src)) {
  console.warn(`[copy-mediainfo-wasm] source not found, skipping: ${src}`);
  process.exit(0);
}

mkdirSync(destDir, { recursive: true });
copyFileSync(src, dest);
console.log(`[copy-mediainfo-wasm] copied to ${dest}`);
