import { useTranslation } from "react-i18next";
import { format } from "date-fns";
import { uk } from "date-fns/locale";
import { AlertTriangle, Satellite } from "lucide-react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { useQuota } from "@/hooks/useQuota";
import { cn } from "@/lib/utils";

export function QuotaCard() {
  const { t } = useTranslation();
  const { data: quota, isLoading } = useQuota();

  if (isLoading || !quota) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Satellite className="h-4 w-4 text-primary" />
            {t("quota.title")}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-2 w-full animate-pulse rounded-full bg-muted" />
        </CardContent>
      </Card>
    );
  }

  const pct = quota.percent_used;
  const status: "ok" | "warn" | "danger" =
    pct < 60 ? "ok" : pct < 85 ? "warn" : "danger";

  const barClass = {
    ok: "[&>div]:bg-emerald-500",
    warn: "[&>div]:bg-amber-500",
    danger: "[&>div]:bg-red-500",
  }[status];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Satellite className="h-4 w-4 text-primary" />
          {t("quota.title")}
        </CardTitle>
        <CardDescription>{t("quota.tooltip")}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-3 gap-3 text-sm">
          <div>
            <div className="text-xs text-muted-foreground">{t("quota.used")}</div>
            <div className="font-semibold">{quota.units_used.toFixed(2)} PU</div>
          </div>
          <div>
            <div className="text-xs text-muted-foreground">{t("quota.remaining")}</div>
            <div className="font-semibold">{quota.units_remaining.toFixed(2)} PU</div>
          </div>
          <div>
            <div className="text-xs text-muted-foreground">{t("quota.limit")}</div>
            <div className="font-semibold">{quota.units_limit.toFixed(0)} PU</div>
          </div>
        </div>

        <div>
          <Progress value={pct} className={cn("h-3", barClass)} />
          <div className="mt-1 text-right text-xs text-muted-foreground">
            {pct.toFixed(1)}%
          </div>
        </div>

        <div className="text-xs text-muted-foreground">
          {t("quota.resetsAt")}: {format(new Date(quota.refresh_at), "d MMMM yyyy", { locale: uk })}
        </div>

        {status === "danger" && (
          <div className="flex items-start gap-2 rounded-md border border-red-200 bg-red-50 p-2 text-xs text-red-700">
            <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0" />
            <span>{pct >= 100 ? t("quota.exhausted") : t("quota.almostFull")}</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
