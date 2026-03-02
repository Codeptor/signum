"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { AUTO_REFRESH_INTERVAL } from "@/lib/constants";

export interface UseAutoRefreshReturn {
  refreshIn: number;
  paused: boolean;
  setPaused: (paused: boolean) => void;
  resetTimer: () => void;
}

export function useAutoRefresh(
  onRefresh: () => void,
  interval: number = AUTO_REFRESH_INTERVAL
): UseAutoRefreshReturn {
  const [refreshIn, setRefreshIn] = useState(30);
  const [paused, setPausedState] = useState(false);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const countdownRef = useRef<NodeJS.Timeout | null>(null);

  const setPaused = useCallback((value: boolean) => {
    setPausedState(value);
  }, []);

  const resetTimer = useCallback(() => {
    setRefreshIn(30);
  }, []);

  // 1-second countdown tick
  useEffect(() => {
    countdownRef.current = setInterval(() => {
      setRefreshIn((prev) => (prev <= 1 ? 30 : prev - 1));
    }, 1000);

    return () => {
      if (countdownRef.current) {
        clearInterval(countdownRef.current);
      }
    };
  }, []);

  // Auto-refresh interval
  useEffect(() => {
    intervalRef.current = setInterval(() => {
      if (!paused) {
        onRefresh();
      }
    }, interval);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [onRefresh, interval, paused]);

  return { refreshIn, paused, setPaused, resetTimer };
}
