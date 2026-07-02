// Security headers applied to every response. These four are safe for a
// Next.js app and never break runtime. A Content-Security-Policy is
// intentionally NOT set here yet: a strict CSP on Next.js requires per-request
// nonces for its inline runtime scripts/styles and must be validated in a real
// browser first (see security/plans/SECURITY_HEADERS_PLAN.md). Add it once
// verified so it doesn't break the app.
const securityHeaders = [
  { key: 'Strict-Transport-Security', value: 'max-age=63072000; includeSubDomains; preload' },
  { key: 'X-Frame-Options', value: 'DENY' },
  { key: 'X-Content-Type-Options', value: 'nosniff' },
  { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
]

/** @type {import('next').NextConfig} */
const nextConfig = {
  async headers() {
    return [{ source: '/:path*', headers: securityHeaders }]
  },
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'images.unsplash.com',
        pathname: '/**',
      },
      {
        protocol: 'https',
        hostname: 'unsplash.com',
        pathname: '/**',
      },
    ],
  },
}

export default nextConfig
