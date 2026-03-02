"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { AUTO_REFRESH_INTERVAL } from "@/lib/constants";

export interface UseAutoRefreshReturn {
  refreshIn: number;
  paused: boolean;
  setPaused: (paused: boolean) => void;
  togglePause: () => void;
  resetTimer: () => void;
}

export function useAutoRefresh(
  onRefresh: () => void,
  interval: number = AUTO_REFRESH_INTERVAL
): UseAutoRefreshReturn {
  const totalSeconds = Math.round(interval / 1000);
  const [refreshIn, setRefreshIn] = useState(totalSeconds);
  const [paused, setPausedState] = useState(false);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const countdownRef = useRef<NodeJS.Timeout | null>(null);
  // Stable ref so the interval closure never captures a stale value
  const pausedRef = useRef(paused);

  const setPaused = useCallback((value: boolean) => {
    pausedRef.current = value;
    setPausedState(value);
  }, []);

  const togglePause = useCallback(() => {
    setPaused(!pausedRef.current);
  }, [setPaused]);

  const resetTimer = useCallback(() => {
    setRefreshIn(totalSeconds);
  }, [totalSeconds]);

  // 1-second countdown tick
  useEffect(() => {
    countdownRef.current = setInterval(() => {
      setRefreshIn((prev) => (prev <= 1 ? totalSeconds : prev - 1));
    }, 1000);

    return () => {
      if (countdownRef.current) {
        clearInterval(countdownRef.current);
      }
    };
  }, [totalSeconds]);

  // Auto-refresh interval — stable, never recreated on pause toggle
  useEffect(() => {
    intervalRef.current = setInterval(() => {
      if (!pausedRef.current) {
        onRefresh();
      }
    }, interval);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [onRefresh, interval]); // paused intentionally excluded — read via pausedRef

  return { refreshIn, paused, setPaused, togglePause, resetTimer };
}
