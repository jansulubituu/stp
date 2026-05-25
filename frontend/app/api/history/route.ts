import { NextRequest, NextResponse } from "next/server";

/**
 * Custom API route handler cho /api/history.
 * Bypass Next.js rewrite proxy để kiểm soát timeout.
 */

export async function GET() {
  try {
    const backendRes = await fetch("http://localhost:8000/api/history", {
      headers: { "Content-Type": "application/json" },
    });

    const data = await backendRes.json();
    return NextResponse.json(data, { status: backendRes.status });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Unknown error";
    console.error("[API Route /api/history] Backend call failed:", message);
    return NextResponse.json(
      { detail: `Lỗi kết nối Backend: ${message}` },
      { status: 502 }
    );
  }
}
