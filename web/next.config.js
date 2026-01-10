/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Proxy API requests to FastAPI backend
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
    ]
  },
  async redirects() {
    return [
      {
        source: '/diagnostics',
        destination: '/workforce-health',
        permanent: true,
      },
      {
        source: '/future-radar',
        destination: '/flight-risk',
        permanent: true,
      },
      {
        source: '/survival',
        destination: '/retention-forecast',
        permanent: true,
      },
    ]
  },
}

module.exports = nextConfig
