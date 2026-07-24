/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  webpack(config) {
    // mediainfo.js's bundled build has a `new URL('MediaInfoModule.wasm', import.meta.url)`
    // fallback. We self-host the WASM in public/ and load it via locateFile, so that fallback
    // is dead code at runtime — tell webpack not to try to resolve it as a bundled asset.
    config.module.rules.push({
      test: /[\\/]mediainfo\.js[\\/]dist[\\/]/,
      parser: { url: false },
    });
    return config;
  },
};

export default nextConfig;
