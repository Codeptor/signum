"use client";

import { useState, useCallback } from "react";
import { fetchStatus, fetchHealth, fetchEquity } from "@/lib/api";
import { storeSessionPoint } from "@/lib/api";
import { StatusData, EquityPoint, MarketSession, SessionPoint } from "@/types/dashboard";
import { SESSION_POINT_LIMIT } from "@/lib/constants";
import { getMarketStatus } from "@/lib/market-utils";

export interface ComparisonDataState {
  statusA: StatusData | null;
  statusB: StatusData | null;
  healthA: boolean | null;
  healthB: boolean | null;
  equityA: EquityPoint[];
  equityB: EquityPoint[];
}

export interface UseComparisonDataReturn extends ComparisonDataState {
  loadComparison: () => Promise<void>;
  loadEquityCurves: () => Promise<void>;
  checkAndNotifyRegimeChange: (prevRegimeA: string | null, prevRegimeB: string | null) => 
    { newRegimeA: string | null; newRegimeB: string | null };
  accumulateSessionPoints: (currentPoints: SessionPoint[]) => SessionPoint[];
}

export function useComparisonData(
  onNotify: (message: string) => void
): UseComparisonDataReturn {
  const [state, setState] = useState<ComparisonDataState>({
    statusA: null,
    statusB: null,
    healthA: null,
    healthB: null,
    equityA: [],
    equityB: [],
  });

  const loadComparison = useCallback(async () => {
    const [sA, sB, hA, hB] = await Promise.all([
      fetchStatus("bot-a"),
      fetchStatus("bot-b"),
      fetchHealth("bot-a"),
      fetchHealth("bot-b"),
    ]);

    setState((prev) => ({
      ...prev,
      statusA: sA,
      statusB: sB,
      healthA: hA != null,
      healthB: hB != null,
    }));
  }, []);

  const loadEquityCurves = useCallback(async () => {
    const [eA, eB] = await Promise.all([
      fetchEquity("bot-a"),
      fetchEquity("bot-b"),
    ]);

    setState((prev) => ({
      ...prev,
      equityA: eA,
      equityB: eB,
    }));
  }, []);

  const checkAndNotifyRegimeChange = useCallback(
    (prevRegimeA: string | null, prevRegimeB: string | null) => {
      const regimeA = state.statusA?.regime?.regime;
      const regimeB = state.statusB?.regime?.regime;

      if (
        prevRegimeA &&
        regimeA &&
        regimeA !== prevRegimeA &&
        regimeA !== "normal"
      ) {
        onNotify(`Bot A regime: ${regimeA.toUpperCase()}`);
      }
      if (
        prevRegimeB &&
        regimeB &&
        regimeB !== prevRegimeB &&
        regimeB !== "normal"
      ) {
        onNotify(`Bot B regime: ${regimeB.toUpperCase()}`);
      }

      return { newRegimeA: regimeA ?? null, newRegimeB: regimeB ?? null };
    },
    [state.statusA?.regime?.regime, state.statusB?.regime?.regime, onNotify]
  );

  const accumulateSessionPoints = useCallback(
    (currentPoints: SessionPoint[]): SessionPoint[] => {
      const market = getMarketStatus(new Date());
      if (market.session !== "Open" as MarketSession) {
        return currentPoints;
      }

      const eqA = state.statusA?.account?.equity;
      const eqB = state.statusB?.account?.equity;

      if (eqA == null && eqB == null) {
        return currentPoints;
      }

      const now = new Date();
      const time = now.toLocaleTimeString("en-US", {
        timeZone: "America/New_York",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false,
      });

      const spyDd =
        state.statusA?.regime?.spy_drawdown ??
        state.statusB?.regime?.spy_drawdown ??
        null;
      const posA = state.statusA?.positions_count ?? null;
      const posB = state.statusB?.positions_count ?? null;

      // Persist to Postgres (fire-and-forget)
      storeSessionPoint(
        eqA != null ? +eqA.toFixed(2) : null,
        eqB != null ? +eqB.toFixed(2) : null,
        now.toISOString(),
        spyDd != null ? +spyDd.toFixed(6) : null,
        posA,
        posB
      );

      return [
        ...currentPoints,
        {
          time,
          a: eqA != null ? +eqA.toFixed(2) : null,
          b: eqB != null ? +eqB.toFixed(2) : null,
          spy_dd: spyDd != null ? +spyDd.toFixed(6) : null,
          pos_a: posA,
          pos_b: posB,
        },
      ].slice(-SESSION_POINT_LIMIT);
    },
    [state.statusA, state.statusB]
  );

  return {
    ...state,
    loadComparison,
    loadEquityCurves,
    checkAndNotifyRegimeChange,
    accumulateSessionPoints,
  };
}
