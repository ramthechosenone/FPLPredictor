import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  // Rewrites only work in dev mode (ignored for static export)
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "https://fpl-predictor-176753204897.us-central1.run.app/:path*",
      },
    ];
  },
};

export default nextConfig;
