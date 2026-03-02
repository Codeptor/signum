"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ShieldCheckIcon } from "@/components/ui/shield-check";
import { RiskData } from "@/types/dashboard";
import { fmt, fmtPct } from "@/lib/formatters";
import { MetricRow } from "./MetricRow";

interface RiskPanelProps {
  risk: RiskData | null;
}

export function RiskPanel({ risk }: RiskPanelProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-1.5">
          <ShieldCheckIcon size={16} />
          Risk Metrics
        </CardTitle>
      </CardHeader>
      <CardContent>
        <dl className="space-y-3 text-xs">
          <MetricRow label="Sharpe Ratio" value={fmt(risk?.sharpe_ratio)} />
          <MetricRow label="Sortino Ratio" value={fmt(risk?.sortino_ratio)} />
          <MetricRow label="Max Drawdown" value={fmtPct(risk?.max_drawdown)} />
          <MetricRow label="Current DD" value={fmtPct(risk?.current_drawdown)} />
          <MetricRow label="VaR 95%" value={fmtPct(risk?.var_95)} />
          <MetricRow label="CVaR 95%" value={fmtPct(risk?.cvar_95)} />
          <MetricRow label="Win Rate" value={fmtPct(risk?.win_rate)} />
          <MetricRow
            label="Total Trades"
            value={String(risk?.total_trades ?? "—")}
          />
        </dl>
      </CardContent>
    </Card>
  );
}
