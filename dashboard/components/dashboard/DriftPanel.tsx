"use client";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { WavesIcon } from "@/components/ui/waves";
import { DriftData } from "@/types/dashboard";
import { MetricRow } from "./MetricRow";

interface DriftPanelProps {
  drift: DriftData | null;
}

export function DriftPanel({ drift }: DriftPanelProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-1.5">
          <WavesIcon size={16} />
          Feature Drift
        </CardTitle>
        <CardDescription>KS test + PSI monitoring</CardDescription>
      </CardHeader>
      <CardContent>
        <dl className="space-y-3 text-xs">
          <MetricRow
            label="Drifted"
            value={`${drift?.drift_count ?? "—"} / ${drift?.total_features ?? "—"}`}
          />
        </dl>
        {drift?.drifted_features && drift.drifted_features.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1">
            {drift.drifted_features.map((f) => (
              <Badge key={f} variant="secondary" className="text-[10px]">
                {f}
              </Badge>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
