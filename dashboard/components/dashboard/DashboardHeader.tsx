"use client";

import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { SparklesIcon } from "@/components/ui/sparkles";
import { ClockIcon } from "@/components/ui/clock";
import { RefreshCWIcon } from "@/components/ui/refresh-cw";
import { PauseIcon } from "@/components/ui/pause";
import { BotId, MarketSession } from "@/types/dashboard";
import { fmtTz } from "@/lib/formatters";
import { getSessionVariant } from "@/lib/market-utils";

interface DashboardHeaderProps {
  bot: BotId;
  onSwitchBot: (bot: BotId) => void;
  marketSession: MarketSession;
  marketCountdown: string;
  now: Date;
  refreshIn: number;
  isPaused: boolean;
}

export function DashboardHeader({
  bot,
  onSwitchBot,
  marketSession,
  marketCountdown,
  now,
  refreshIn,
  isPaused,
}: DashboardHeaderProps) {
  return (
    <header className="border-b border-border px-6 py-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <SparklesIcon size={14} />
            <h1 className="text-sm font-semibold tracking-tight">SIGNUM</h1>
          </div>
          <Separator orientation="vertical" className="h-4" />
          <Badge variant={getSessionVariant(marketSession)}>
            {marketSession}
          </Badge>
          <span className="text-xs tabular-nums text-muted-foreground">
            {marketSession === "Open"
              ? `Closes in ${marketCountdown}`
              : `Opens in ${marketCountdown}`}
          </span>
        </div>

        <div className="flex items-center gap-4">
          <Tabs
            value={bot}
            onValueChange={(value) => onSwitchBot(value as BotId)}
          >
            <TabsList variant="line">
              <TabsTrigger value="bot-a">Bot A</TabsTrigger>
              <TabsTrigger value="bot-b">Bot B</TabsTrigger>
            </TabsList>
          </Tabs>

          <Separator orientation="vertical" className="h-4" />

          <div className="flex items-center gap-3 text-[10px] tabular-nums text-muted-foreground">
            <ClockIcon size={12} />
            <span>{fmtTz(now, "America/New_York", "NY")}</span>
            <span>{fmtTz(now, "Asia/Kolkata", "IST")}</span>
            <span>{fmtTz(now, "UTC", "UTC")}</span>
          </div>

          <Separator orientation="vertical" className="h-4" />

          <div className="flex items-center gap-1 text-[10px] tabular-nums text-muted-foreground">
            {isPaused ? (
              <PauseIcon size={12} />
            ) : (
              <RefreshCWIcon size={12} />
            )}
            <span>{isPaused ? "paused" : `${refreshIn}s`}</span>
          </div>
        </div>
      </div>
    </header>
  );
}
