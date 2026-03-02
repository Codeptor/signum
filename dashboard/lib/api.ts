import { BotId, StatusData, Position, RiskData, TcaData, DriftData, EquityPoint, HealthData } from "./types";

const BASE = "/api/bot";

async function fetchBot<T>(bot: BotId, endpoint: string): Promise<T | null> {
  try {
    const res = await fetch(`${BASE}/${bot}/${endpoint}`, {
      cache: "no-store",
    });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function fetchStatus(bot: BotId) {
  return fetchBot<StatusData>(bot, "api/status");
}

export async function fetchPositions(bot: BotId): Promise<Position[]> {
  const data = await fetchBot<{ positions: Position[] } | Position[]>(bot, "api/positions");
  if (!data) return [];
  if (Array.isArray(data)) return data;
  if (Array.isArray(data.positions)) return data.positions;
  return [];
}

export async function fetchRisk(bot: BotId) {
  return fetchBot<RiskData>(bot, "api/risk");
}

export async function fetchTca(bot: BotId) {
  return fetchBot<TcaData>(bot, "api/tca");
}

export async function fetchDrift(bot: BotId) {
  return fetchBot<DriftData>(bot, "api/drift");
}

export async function fetchEquity(bot: BotId): Promise<EquityPoint[]> {
  type RawPoint = { date?: string; timestamp?: string; equity: number };
  const data = await fetchBot<
    RawPoint[] |
    { history: RawPoint[] } |
    { equity: RawPoint[] } |
    { records: RawPoint[] }
  >(bot, "api/equity");
  if (!data) return [];

  // Unwrap whichever envelope key the backend uses
  let raw: RawPoint[] = [];
  if (Array.isArray(data))                                    raw = data;
  else if ("history" in data && Array.isArray(data.history)) raw = data.history;
  else if ("equity"  in data && Array.isArray(data.equity))  raw = data.equity;
  else if ("records" in data && Array.isArray(data.records)) raw = data.records;

  // Normalize: backend sends "date", EquityPoint type expects "timestamp"
  return raw.map((pt) => ({
    timestamp: pt.timestamp ?? pt.date ?? "",
    equity: pt.equity,
  }));
}

export async function fetchHealth(bot: BotId) {
  return fetchBot<HealthData>(bot, "healthz");
}

export async function fetchLogs(bot: BotId, lines = 50): Promise<string> {
  const data = await fetchBot<{ log: string[]; logs: string } | string>(bot, `api/logs?lines=${lines}`);
  if (!data) return "";
  if (typeof data === "string") return data;
  if (Array.isArray(data.log)) return data.log.join("\n");
  if (data.logs) return data.logs;
  return "";
}
