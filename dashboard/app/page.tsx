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
  const [lastRefresh, setLastRefresh] = React.useState<Date | null>(null);

  // ── Hooks ───────────────────────────────────────────────────────────────
  const { notify } = useNotifications();
  const { points: sessionPoints, setPoints: setSessionPoints, clearPoints } = useSessionPoints();
  const { data: botData, loadBot, refresh } = useBotData();
  const comparison = useComparisonData(notify);

  // Destructure stable callbacks so handleRefresh dep array has no object refs
  const { loadComparison, loadEquityCurves, accumulateSessionPoints } = comparison;

  const market = getMarketStatus(now);

  // ── Handlers ────────────────────────────────────────────────────────────

  const handleRefresh = React.useCallback(() => {
    loadComparison();
    loadEquityCurves();
    // Functional updater — no sessionPoints in dep array
    setSessionPoints((prev) => accumulateSessionPoints(prev));
    refresh(bot);
    setLastRefresh(new Date());
  }, [bot, loadComparison, loadEquityCurves, accumulateSessionPoints, refresh, setSessionPoints]);

  const { refreshIn, paused, togglePause } = useAutoRefresh(handleRefresh);

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
    onTogglePause: togglePause,
  });

  // Initial load
  React.useEffect(() => {
    handleRefresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Refresh when bot changes
  React.useEffect(() => {
    loadBot(bot);
  }, [bot, loadBot]);

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
