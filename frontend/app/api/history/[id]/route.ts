import { NextRequest, NextResponse } from "next/server";

/**
 * Custom API route handler cho /api/history/:id (GET, DELETE).
 */

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  try {
    const backendRes = await fetch(`http://localhost:8000/api/history/${id}`, {
      headers: { "Content-Type": "application/json" },
    });
    const data = await backendRes.json();
    return NextResponse.json(data, { status: backendRes.status });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ detail: message }, { status: 502 });
  }
}

export async function DELETE(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  try {
    const backendRes = await fetch(`http://localhost:8000/api/history/${id}`, {
      method: "DELETE",
    });
    return new NextResponse(null, { status: backendRes.status });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ detail: message }, { status: 502 });
  }
}
