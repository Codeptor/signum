"use client";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { ChartBarIncreasingIcon } from "@/components/ui/chart-bar-increasing";
import { Position } from "@/types/dashboard";
import { SECTOR_MAP } from "@/lib/constants";

interface SectorExposureProps {
  positions: Position[];
}

export function SectorExposure({ positions }: SectorExposureProps) {
  const totalValue = positions.reduce(
    (s, p) => s + Math.abs(p.market_value ?? 0),
    0
  );

  if (totalValue === 0) return null;

  // Group by sector
  const sectorWeights = new Map<string, number>();
  for (const p of positions) {
    const sector = SECTOR_MAP[p.symbol] || "Other";
    sectorWeights.set(
      sector,
      (sectorWeights.get(sector) || 0) + Math.abs(p.market_value ?? 0)
    );
  }

  // Sort descending by weight
  const sorted = Array.from(sectorWeights.entries())
    .map(([sector, value]) => ({ sector, weight: value / totalValue }))
    .sort((a, b) => b.weight - a.weight);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-1.5">
          <ChartBarIncreasingIcon size={16} />
          Sector Exposure
        </CardTitle>
        <CardDescription>Portfolio weight by sector</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {sorted.map(({ sector, weight }) => (
            <div key={sector} className="flex items-center gap-3 text-xs">
              <span className="w-28 shrink-0 text-muted-foreground">
                {sector}
              </span>
              <div className="flex-1 h-2 bg-muted overflow-hidden">
                <div
                  className="h-full bg-foreground/70"
                  style={{ width: `${(weight * 100).toFixed(1)}%` }}
                />
              </div>
              <span className="w-12 text-right tabular-nums font-medium">
                {(weight * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
