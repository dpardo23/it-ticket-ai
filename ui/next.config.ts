import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  distDir: "out",
  allowedDevOrigins: ["192.168.100.23", "10.187.75.1", "localhost", "127.0.0.1"],
};

export default nextConfig;