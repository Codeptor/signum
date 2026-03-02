"use client";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { TerminalIcon } from "@/components/ui/terminal";

interface LogsPanelProps {
  logs: string;
}

export function LogsPanel({ logs }: LogsPanelProps) {
  return (
    <Card className="col-span-2">
      <CardHeader>
        <CardTitle className="flex items-center gap-1.5">
          <TerminalIcon size={16} />
          Logs
        </CardTitle>
        <CardDescription>Recent bot output</CardDescription>
      </CardHeader>
      <CardContent className="p-0">
        <ScrollArea className="h-72">
          <pre className="p-4 text-[10px] leading-relaxed text-muted-foreground">
            {logs || "No logs available"}
          </pre>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
