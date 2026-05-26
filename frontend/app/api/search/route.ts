import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 120_000);

    const backendRes = await fetch("http://localhost:8000/api/search", {
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
    console.error("[API Route /api/search] Backend call failed:", message);

    if (message.includes("aborted")) {
      return NextResponse.json(
        { detail: "Backend xử lý quá lâu (>120 giây)." },
        { status: 504 }
      );
    }

    return NextResponse.json(
      { detail: `Lỗi kết nối Backend: ${message}` },
      { status: 502 }
    );
  }
}
