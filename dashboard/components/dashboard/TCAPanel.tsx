"use client";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { DollarSignIcon } from "@/components/ui/dollar-sign";
import { TcaData } from "@/types/dashboard";
import { fmt, fmtPct } from "@/lib/formatters";
import { MetricRow } from "./MetricRow";

interface TCAPanelProps {
  tca: TcaData | null;
}

export function TCAPanel({ tca }: TCAPanelProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-1.5">
          <DollarSignIcon size={16} />
          TCA
        </CardTitle>
        <CardDescription>Transaction cost analysis</CardDescription>
      </CardHeader>
      <CardContent>
        <dl className="space-y-3 text-xs">
          <MetricRow
            label="Avg IS (bps)"
            value={fmt(tca?.avg_implementation_shortfall_bps, 1)}
          />
          <MetricRow label="Fill Rate" value={fmtPct(tca?.avg_fill_rate)} />
          <MetricRow
            label="Total Trades"
            value={String(tca?.total_trades ?? "—")}
          />
        </dl>
      </CardContent>
    </Card>
  );
}
