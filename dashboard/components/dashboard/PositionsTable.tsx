"use client";

import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableFooter,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { LayersIcon } from "@/components/ui/layers";
import { Position } from "@/types/dashboard";
import { fmtUsd, fmtPct } from "@/lib/formatters";

interface PositionsTableProps {
  positions: Position[];
}

export function PositionsTable({ positions }: PositionsTableProps) {
  const totalQty = positions.reduce((s, p) => s + (p.qty ?? 0), 0);
  const totalValue = positions.reduce(
    (s, p) => s + (p.market_value ?? 0),
    0
  );
  const totalPnL = positions.reduce(
    (s, p) => s + (p.unrealized_pl ?? 0),
    0
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-1.5">
          <LayersIcon size={16} />
          Open Positions
        </CardTitle>
        <CardDescription>
          {positions.length} position{positions.length !== 1 ? "s" : ""}
        </CardDescription>
      </CardHeader>
      <CardContent>
        {positions.length > 0 ? (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Symbol</TableHead>
                <TableHead className="text-right">Qty</TableHead>
                <TableHead className="text-right">Avg Entry</TableHead>
                <TableHead className="text-right">Current</TableHead>
                <TableHead className="text-right">Market Value</TableHead>
                <TableHead className="text-right">P&L</TableHead>
                <TableHead className="text-right">P&L %</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {positions.map((p) => (
                <TableRow key={p.symbol}>
                  <TableCell className="font-medium">{p.symbol}</TableCell>
                  <TableCell className="text-right tabular-nums">
                    {p.qty}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {fmtUsd(p.avg_entry_price)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {fmtUsd(p.current_price)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {fmtUsd(p.market_value)}
                  </TableCell>
                  <TableCell
                    className={`text-right tabular-nums ${
                      (p.unrealized_pl ?? 0) >= 0
                        ? "text-green-500"
                        : "text-red-500"
                    }`}
                  >
                    {fmtUsd(p.unrealized_pl)}
                  </TableCell>
                  <TableCell
                    className={`text-right tabular-nums ${
                      (p.unrealized_plpc ?? 0) >= 0
                        ? "text-green-500"
                        : "text-red-500"
                    }`}
                  >
                    {fmtPct(p.unrealized_plpc)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
            <TableFooter>
              <TableRow>
                <TableCell className="font-medium">Total</TableCell>
                <TableCell className="text-right tabular-nums">
                  {totalQty}
                </TableCell>
                <TableCell />
                <TableCell />
                <TableCell className="text-right tabular-nums font-medium">
                  {fmtUsd(totalValue)}
                </TableCell>
                <TableCell
                  className={`text-right tabular-nums font-medium ${
                    totalPnL >= 0 ? "text-green-500" : "text-red-500"
                  }`}
                >
                  {fmtUsd(totalPnL)}
                </TableCell>
                <TableCell />
              </TableRow>
            </TableFooter>
          </Table>
        ) : (
          <p className="py-8 text-center text-xs text-muted-foreground">
            No open positions
          </p>
        )}
      </CardContent>
    </Card>
  );
}
