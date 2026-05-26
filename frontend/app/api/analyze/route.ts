import { NextRequest, NextResponse } from "next/server";

/**
 * Custom API route handler cho /api/analyze.
 * Bypass Next.js rewrite proxy để kiểm soát timeout tối đa 300 giây.
 * Giải quyết triệt để lỗi ECONNRESET do proxy mặc định timeout quá ngắn.
 */
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();

    // AbortController với timeout 300 giây
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 300_000);

    const backendRes = await fetch("http://localhost:8000/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    const data = await backendRes.json();

    return NextResponse.json(data, { status: backendRes.status });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Unknown error";
    console.error("[API Route /api/analyze] Backend call failed:", message);

    if (message.includes("aborted")) {
      return NextResponse.json(
        { detail: "Backend xử lý quá lâu (>300 giây). Vui lòng thử lại với truy vấn ngắn hơn." },
        { status: 504 }
      );
    }

    return NextResponse.json(
      { detail: `Lỗi kết nối Backend: ${message}` },
      { status: 502 }
    );
  }
}
