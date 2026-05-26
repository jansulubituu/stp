import { NextRequest } from "next/server";

export const dynamic = "force-dynamic";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ runId: string }> }
) {
  const { runId } = await params;
  const backendRes = await fetch(`http://localhost:8000/api/analysis-runs/${runId}/events`, {
    headers: { Accept: "text/event-stream" },
    cache: "no-store",
  });

  return new Response(backendRes.body, {
    status: backendRes.status,
    headers: {
      "Content-Type": backendRes.headers.get("Content-Type") ?? "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}
