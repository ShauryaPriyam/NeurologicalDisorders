import { NextRequest, NextResponse } from "next/server";

/** Proxies MRI upload to the FastAPI `/predict` endpoint (full per-model breakdown). */
export async function POST(request: NextRequest) {
  const formData = await request.formData();
  const base = (
    process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
  ).replace(/\/+$/, "");
  const res = await fetch(`${base}/predict`, {
    method: "POST",
    body: formData,
  });
  let data: unknown = {};
  try {
    data = await res.json();
  } catch {
    data = { detail: "Invalid JSON from analysis server" };
  }
  return NextResponse.json(data, { status: res.status });
}
