import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Không dùng rewrites() nữa - đã chuyển sang Custom API Route Handlers
  // trong app/api/analyze/route.ts và app/api/history/route.ts
  // để kiểm soát trực tiếp timeout (120s) và xử lý lỗi chính xác.
};

export default nextConfig;
