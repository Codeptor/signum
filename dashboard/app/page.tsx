"use client";

import * as React from "react";
import { BotId } from "@/types/dashboard";
import { getMarketStatus } from "@/lib/market-utils";
import { useNotifications } from "@/hooks/useNotifications";
import { useSessionPoints } from "@/hooks/useSessionPoints";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import { useAutoRefresh } from "@/hooks/useAutoRefresh";
import { useBotData } from "@/hooks/useBotData";
import { useComparisonData } from "@/hooks/useComparisonData";

import {
  DashboardHeader,
  ComparisonStrip,
  HeroMetrics,
  DualEquityChart,
  LiveSessionChart,
  SectorExposure,
  PositionsTable,
  RiskPanel,
  LogsPanel,
  TCAPanel,
  DriftPanel,
} from "@/components/dashboard";

export default function DashboardPage() {
  // ── State ───────────────────────────────────────────────────────────────
  const [bot, setBot] = React.useState<BotId>("bot-a");
  const [now, setNow] = React.useState<Date>(new Date());
  const [lastRefresh, setLastRefresh] = React.useState<Date>(new Date());
  const [prevRegimeA, setPrevRegimeA] = React.useState<string | null>(null);
  const [prevRegimeB, setPrevRegimeB] = React.useState<string | null>(null);

  // ── Hooks ───────────────────────────────────────────────────────────────
  const { notify } = useNotifications();
  const { points: sessionPoints, setPoints: setSessionPoints, clearPoints } = useSessionPoints();
  const { data: botData, loadBot, refresh } = useBotData();
  const comparison = useComparisonData(notify);

  const market = getMarketStatus(now);

  // ── Handlers ────────────────────────────────────────────────────────────

  const handleRefresh = React.useCallback(() => {
    // Load comparison data
    comparison.loadComparison().then(() => {
      comparison.loadEquityCurves();

      // Regime change notifications
      const { newRegimeA, newRegimeB } = comparison.checkAndNotifyRegimeChange(
        prevRegimeA,
        prevRegimeB
      );
      if (newRegimeA) setPrevRegimeA(newRegimeA);
      if (newRegimeB) setPrevRegimeB(newRegimeB);

      // Accumulate session points
      const newPoints = comparison.accumulateSessionPoints(sessionPoints);
      if (newPoints.length > sessionPoints.length) {
        setSessionPoints(newPoints);
      }
    });

    // Load current bot data
    refresh(bot);

    setLastRefresh(new Date());
  }, [bot, comparison, prevRegimeA, prevRegimeB, refresh, sessionPoints, setSessionPoints]);

  const { refreshIn, paused, setPaused, resetTimer } = useAutoRefresh(handleRefresh);

  const handleSwitchBot = React.useCallback((newBot: BotId) => {
    setBot(newBot);
  }, []);

  // ── Effects ─────────────────────────────────────────────────────────────

  // Clock tick
  React.useEffect(() => {
    const tick = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(tick);
  }, []);

  // Dynamic page title
  React.useEffect(() => {
    const eq = botData.status?.account?.equity;
    const label = bot === "bot-a" ? "A" : "B";
    document.title = eq != null ? `$${eq.toLocaleString("en-US", { minimumFractionDigits: 2 })} (${label}) | Signum` : "Signum";
  }, [botData.status, bot]);

  // Keyboard shortcuts
  useKeyboardShortcuts({
    onSwitchBotA: () => setBot("bot-a"),
    onSwitchBotB: () => setBot("bot-b"),
    onRefresh: handleRefresh,
    onTogglePause: () => setPaused(!paused),
  });

  // Initial load and auto-refresh
  React.useEffect(() => {
    handleRefresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Refresh when bot changes
  React.useEffect(() => {
    loadBot(bot);
  }, [bot, loadBot]);

  // Reset timer when handleRefresh changes
  React.useEffect(() => {
    resetTimer();
  }, [resetTimer, handleRefresh]);

  // ── Render ──────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-background text-foreground">
      <DashboardHeader
        bot={bot}
        onSwitchBot={handleSwitchBot}
        marketSession={market.session}
        marketCountdown={market.countdown}
        now={now}
        refreshIn={refreshIn}
        isPaused={paused}
      />

      <main className="space-y-4 p-6">
        {/* Comparison Strip */}
        <ComparisonStrip
          botA={{
            label: "Bot A",
            sublabel: "LightGBM",
            status: comparison.statusA,
            healthy: comparison.healthA,
          }}
          botB={{
            label: "Bot B",
            sublabel: "Ensemble",
            status: comparison.statusB,
            healthy: comparison.healthB,
          }}
          activeBot={bot}
          onSelectBot={handleSwitchBot}
        />

        {/* Hero Metrics */}
        <HeroMetrics
          status={botData.status}
          healthy={botData.healthy}
          loading={botData.loading}
        />

        {/* Equity Chart + Risk */}
        <div className="grid grid-cols-4 gap-4">
          <DualEquityChart
            dataA={comparison.equityA}
            dataB={comparison.equityB}
            isPaused={paused}
          />
          <RiskPanel risk={botData.risk} />
        </div>

        {/* Live Session Chart */}
        <LiveSessionChart data={sessionPoints} onClear={clearPoints} />

        {/* Sector Exposure */}
        {botData.positions.length > 0 && (
          <SectorExposure positions={botData.positions} />
        )}

        {/* Positions Table */}
        <PositionsTable positions={botData.positions} />

        {/* Logs + TCA + Drift */}
        <div className="grid grid-cols-4 gap-4">
          <LogsPanel logs={botData.logs} />
          <TCAPanel tca={botData.tca} />
          <DriftPanel drift={botData.drift} />
        </div>

        {/* Footer */}
        <footer className="border-t border-border pt-4 pb-8 text-center text-xs text-muted-foreground">
          Signum — Paper Trading A/B Comparison | Bot A: LightGBM | Bot B:
          Ensemble (LightGBM + CatBoost + RF + Ridge) | Auto-refresh 30s
          {lastRefresh && (
            <span className="ml-2">
              | Last updated: {lastRefresh.toLocaleTimeString()}
            </span>
          )}
        </footer>
      </main>
    </div>
  );
}
