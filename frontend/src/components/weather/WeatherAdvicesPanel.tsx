import { AlertTriangle, Info, ShieldAlert } from "lucide-react";

import type { AdviceSeverity, WeatherAdvice } from "@/api/weather";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface Props {
  advices: WeatherAdvice[];
}

const SEVERITY_META: Record<AdviceSeverity, {
  icon: React.ComponentType<{ className?: string }>;
  border: string;
  bg: string;
  text: string;
  label: string;
}> = {
  info: {
    icon: Info,
    border: "border-l-blue-400",
    bg: "bg-blue-50",
    text: "text-blue-700",
    label: "Інформація",
  },
  warning: {
    icon: AlertTriangle,
    border: "border-l-amber-400",
    bg: "bg-amber-50",
    text: "text-amber-700",
    label: "Увага",
  },
  critical: {
    icon: ShieldAlert,
    border: "border-l-red-500",
    bg: "bg-red-50",
    text: "text-red-700",
    label: "Критично",
  },
};

/**
 * Stack of rule-based agronomic advisories ordered critical → warning → info.
 * Pure presentation — the backend (`build_advices`) does the rule logic.
 */
export function WeatherAdvicesPanel({ advices }: Props) {
  if (advices.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Рекомендації</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground">
          На найближчий тиждень критичних попереджень немає — умови стабільні.
        </CardContent>
      </Card>
    );
  }

  const sorted = [...advices].sort(
    (a, b) =>
      severityRank(b.severity) - severityRank(a.severity),
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Рекомендації</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {sorted.map((a, idx) => {
          const meta = SEVERITY_META[a.severity];
          const Icon = meta.icon;
          return (
            <div
              key={idx}
              className={cn(
                "flex items-start gap-2 rounded-md border-l-4 p-3 text-sm",
                meta.border,
                meta.bg,
              )}
            >
              <Icon className={cn("mt-0.5 h-4 w-4 shrink-0", meta.text)} />
              <div className="min-w-0 flex-1">
                <div className={cn("font-medium", meta.text)}>{a.title}</div>
                <div className="mt-0.5 text-[12px] leading-snug text-foreground/80">
                  {a.detail}
                </div>
              </div>
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}

function severityRank(s: AdviceSeverity): number {
  switch (s) {
    case "critical":
      return 3;
    case "warning":
      return 2;
    case "info":
    default:
      return 1;
  }
}
