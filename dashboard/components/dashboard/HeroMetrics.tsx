"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { DollarSignIcon } from "@/components/ui/dollar-sign";
import { GaugeIcon } from "@/components/ui/gauge";
import { ActivityIcon } from "@/components/ui/activity";
import { StatusData } from "@/types/dashboard";
import { fmt, fmtUsd, fmtPct, fmtDate } from "@/lib/formatters";
import { getRegimeVariant } from "@/lib/market-utils";

interface HeroMetricsProps {
  status: StatusData | null;
  healthy: boolean | null;
  loading: boolean;
}

export function HeroMetrics({ status, healthy, loading }: HeroMetricsProps) {
  if (loading && !status) {
    return (
      <div className="grid grid-cols-4 gap-4">
        <Card className="col-span-2">
          <CardHeader>
            <Skeleton className="h-3 w-24" />
          </CardHeader>
          <CardContent>
            <Skeleton className="h-6 w-40" />
          </CardContent>
        </Card>
        {[...Array(2)].map((_, i) => (
          <Card key={i}>
            <CardHeader>
              <Skeleton className="h-3 w-24" />
            </CardHeader>
            <CardContent>
              <Skeleton className="h-6 w-32" />
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-4 gap-4">
      {/* Portfolio Equity */}
      <Card className="col-span-2">
        <CardHeader>
          <CardDescription className="flex items-center gap-1.5">
            <DollarSignIcon size={14} />
            Portfolio Equity
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-baseline justify-between">
            <CardTitle className="text-2xl font-semibold tabular-nums">
              {fmtUsd(status?.account?.equity)}
            </CardTitle>
            <div className="text-right text-xs text-muted-foreground">
              <p>Cash: {fmtUsd(status?.account?.cash)}</p>
              <p>Buying Power: {fmtUsd(status?.account?.buying_power)}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Market Regime */}
      <Card>
        <CardHeader>
          <CardDescription className="flex items-center gap-1.5">
            <GaugeIcon size={14} />
            Market Regime
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2">
            <Badge
              variant={getRegimeVariant(status?.regime?.regime) as "outline" | "secondary" | "destructive"}
            >
              {status?.regime?.regime?.toUpperCase() || "—"}
            </Badge>
            <span className="text-xs text-muted-foreground">
              Exposure: {fmtPct(status?.regime?.exposure_multiplier)}
            </span>
          </div>
          <p className="mt-2 text-xs text-muted-foreground">
            VIX: {fmt(status?.regime?.vix, 1)} | SPY DD:{" "}
            {fmtPct(status?.regime?.spy_drawdown)}
          </p>
        </CardContent>
      </Card>

      {/* Bot State */}
      <Card>
        <CardHeader>
          <CardDescription className="flex items-center gap-1.5">
            <ActivityIcon size={14} />
            Bot State
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2">
            <span
              className={`inline-block h-2 w-2 rounded-full ${
                healthy ? "bg-green-500" : "bg-red-500"
              }`}
            />
            <CardTitle className="text-sm">
              {healthy ? "Online" : "Offline"}
            </CardTitle>
          </div>
          <p className="mt-2 text-xs text-muted-foreground">
            Positions: {status?.positions_count ?? 0} | Last:{" "}
            {fmtDate(status?.bot_state?.last_shutdown)}
          </p>
          <p className="text-xs text-muted-foreground">
            Reason: {status?.bot_state?.reason || "—"}
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
