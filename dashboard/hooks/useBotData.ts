"use client";

import { useState, useCallback } from "react";
import {
  fetchStatus,
  fetchPositions,
  fetchRisk,
  fetchTca,
  fetchDrift,
  fetchEquity,
  fetchLogs,
  fetchHealth,
} from "@/lib/api";
import {
  BotId,
  StatusData,
  Position,
  RiskData,
  TcaData,
  DriftData,
  EquityPoint,
} from "@/types/dashboard";
import { LOG_LINES_DEFAULT } from "@/lib/constants";

export interface BotData {
  status: StatusData | null;
  positions: Position[];
  risk: RiskData | null;
  tca: TcaData | null;
  drift: DriftData | null;
  equity: EquityPoint[];
  logs: string;
  healthy: boolean | null;
  loading: boolean;
}

export interface UseBotDataReturn {
  data: BotData;
  loadBot: (bot: BotId) => Promise<void>;
  refresh: (bot: BotId) => Promise<void>;
}

export function useBotData(): UseBotDataReturn {
  const [data, setData] = useState<BotData>({
    status: null,
    positions: [],
    risk: null,
    tca: null,
    drift: null,
    equity: [],
    logs: "",
    healthy: null,
    loading: true,
  });

  const loadBot = useCallback(async (bot: BotId) => {
    setData((prev) => ({ ...prev, loading: true }));

    try {
      const [status, positions, risk, tca, drift, equity, logs, health] = await Promise.all([
        fetchStatus(bot),
        fetchPositions(bot),
        fetchRisk(bot),
        fetchTca(bot),
        fetchDrift(bot),
        fetchEquity(bot),
        fetchLogs(bot, LOG_LINES_DEFAULT),
        fetchHealth(bot),
      ]);

      setData({
        status,
        positions: Array.isArray(positions) ? positions : [],
        risk,
        tca,
        drift,
        equity: Array.isArray(equity) ? equity : [],
        logs: logs || "",
        healthy: health != null,
        loading: false,
      });
    } catch {
      setData((prev) => ({ ...prev, loading: false }));
    }
  }, []);

  const refresh = useCallback(async (bot: BotId) => {
    await loadBot(bot);
  }, [loadBot]);

  return { data, loadBot, refresh };
}
