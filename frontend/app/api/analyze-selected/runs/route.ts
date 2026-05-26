import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const backendRes = await fetch("http://localhost:8000/api/analyze-selected/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await backendRes.json();
    return NextResponse.json(data, { status: backendRes.status });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Lỗi không xác định";
    return NextResponse.json({ detail: `Lỗi kết nối Backend: ${message}` }, { status: 502 });
  }
}
