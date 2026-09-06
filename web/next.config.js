/** @type {import('next').NextConfig} */
const desktopBuild = process.env.PEOPLEOS_DESKTOP_BUILD === '1'

const webServerRoutes = {
  // Browser development keeps the convenient FastAPI proxy. Desktop builds
  // are static and run on the same origin as the bundled FastAPI server.
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

const nextConfig = {
  reactStrictMode: true,
  ...(desktopBuild
    ? { output: 'export', trailingSlash: true }
    : webServerRoutes),
}

module.exports = nextConfig
