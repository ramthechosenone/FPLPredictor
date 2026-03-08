import { PredictionsResponse, HealthResponse } from "./types";

// In dev mode, Next.js rewrites /api/* to the Cloud Run backend (avoids CORS).
// In production (static export), calls go directly to the API.
const API_URL =
  typeof window !== "undefined" && window.location.hostname === "localhost"
    ? "/api"
    : process.env.NEXT_PUBLIC_API_URL;

export async function fetchTopPredictions(
  n: number = 100
): Promise<PredictionsResponse> {
  const res = await fetch(`${API_URL}/predict/top?n=${n}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_URL}/health`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}
