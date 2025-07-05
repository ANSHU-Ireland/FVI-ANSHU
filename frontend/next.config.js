/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  experimental: {
    appDir: true,
  },
  env: {
    API_BASE_URL: process.env.API_BASE_URL || 'http://localhost:8000',
    VECTOR_RAG_URL: process.env.VECTOR_RAG_URL || 'http://localhost:8001',
    JAEGER_ENDPOINT: process.env.JAEGER_ENDPOINT || 'http://localhost:14268',
    PROMETHEUS_ENDPOINT: process.env.PROMETHEUS_ENDPOINT || 'http://localhost:9090',
  },
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block',
          },
        ],
      },
    ];
  },
};

module.exports = nextConfig;
