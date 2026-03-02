"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { BotIcon } from "@/components/ui/bot";
import { TrendingUpIcon } from "@/components/ui/trending-up";
import { TrendingDownIcon } from "@/components/ui/trending-down";
import { BotId, StatusData } from "@/types/dashboard";
import { fmtUsd, fmtPct, fmt } from "@/lib/formatters";
import { STARTING_EQUITY } from "@/lib/constants";
import { getRegimeVariant } from "@/lib/market-utils";

interface ComparisonCardData {
  label: string;
  sublabel: string;
  status: StatusData | null;
  healthy: boolean | null;
}

interface ComparisonStripProps {
  botA: ComparisonCardData;
  botB: ComparisonCardData;
  activeBot: BotId;
  onSelectBot: (bot: BotId) => void;
}

function ComparisonCard({
  label,
  sublabel,
  status,
  healthy,
  active,
  onClick,
}: ComparisonCardData & { active: boolean; onClick: () => void }) {
  const equity = status?.account?.equity;
  const equityDelta = equity != null ? equity - STARTING_EQUITY : null;
  const equityPct =
    equity != null ? ((equity - STARTING_EQUITY) / STARTING_EQUITY) * 100 : null;

  return (
    <Card
      className={`cursor-pointer transition-colors ${
        active ? "ring-1 ring-foreground/30" : "opacity-60 hover:opacity-80"
      }`}
      onClick={onClick}
    >
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span
              className={`inline-block h-2 w-2 rounded-full ${
                healthy
                  ? "bg-green-500"
                  : healthy === false
                    ? "bg-red-500"
                    : "bg-muted"
              }`}
            />
            <BotIcon size={16} />
            <CardTitle>{label}</CardTitle>
            <Badge variant="outline" className="text-[10px]">
              {sublabel}
            </Badge>
          </div>
          <Badge
            variant={getRegimeVariant(status?.regime?.regime) as "outline" | "secondary" | "destructive"}
          >
            {status?.regime?.regime?.toUpperCase() || "—"}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex items-baseline justify-between">
          <div>
            <p className="text-lg font-semibold tabular-nums">
              {fmtUsd(equity)}
            </p>
            <div className="flex items-center gap-2">
              <p className="text-xs text-muted-foreground">
                {status?.positions_count ?? 0} positions
              </p>
              {equityDelta != null && (
                <span
                  className={`flex items-center gap-1 text-xs font-medium tabular-nums ${
                    equityDelta >= 0 ? "text-green-500" : "text-red-500"
                  }`}
                >
                  {equityDelta >= 0 ? (
                    <TrendingUpIcon size={12} />
                  ) : (
                    <TrendingDownIcon size={12} />
                  )}
                  {equityDelta >= 0 ? "+" : ""}
                  {fmtUsd(equityDelta)} ({equityDelta >= 0 ? "+" : ""}
                  {fmt(equityPct)}%)
                </span>
              )}
            </div>
          </div>
          <div className="text-right text-xs text-muted-foreground">
            <p>VIX: {fmt(status?.regime?.vix, 1)}</p>
            <p>SPY DD: {fmtPct(status?.regime?.spy_drawdown)}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function ComparisonStrip({
  botA,
  botB,
  activeBot,
  onSelectBot,
}: ComparisonStripProps) {
  return (
    <div className="grid grid-cols-2 gap-4">
      <ComparisonCard
        {...botA}
        active={activeBot === "bot-a"}
        onClick={() => onSelectBot("bot-a")}
      />
      <ComparisonCard
        {...botB}
        active={activeBot === "bot-b"}
        onClick={() => onSelectBot("bot-b")}
      />
    </div>
  );
}
