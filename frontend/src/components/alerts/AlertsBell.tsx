import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";
import { format } from "date-fns";
import { uk } from "date-fns/locale";
import {
  AlertTriangle,
  Bell,
  Check,
  Droplets,
  Flame,
  Loader2,
  TrendingDown,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { AlertRead, AlertSeverity, AlertType } from "@/api/alerts";
import {
  useAcknowledgeAlert,
  useAlerts,
  useAlertsCount,
  useAlertsStream,
} from "@/hooks/useAlerts";

const TYPE_ICONS: Record<AlertType, React.ComponentType<{ className?: string }>> = {
  ndvi_drop: TrendingDown,
  drought_stress: Droplets,
  heat_stress: Flame,
};

const SEVERITY_COLORS: Record<AlertSeverity, string> = {
  info: "text-blue-600",
  warning: "text-amber-600",
  critical: "text-red-600",
};

export function AlertsBell() {
  const { t } = useTranslation();
  // Subscribe to SSE for live push (no-op if not auth'd).
  useAlertsStream();
  const { data: countData } = useAlertsCount();
  const unread = countData?.unread ?? 0;

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon" className="relative h-9 w-9" aria-label={t("alertsUi.bellTitle")}>
          <Bell className="h-5 w-5" />
          {unread > 0 && (
            <span
              className={cn(
                "absolute -right-0.5 -top-0.5 flex h-4 min-w-4 items-center justify-center rounded-full px-1 text-[10px] font-bold text-white",
                unread >= 5 ? "bg-red-600" : "bg-amber-500",
              )}
            >
              {unread > 9 ? "9+" : unread}
            </span>
          )}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuLabel>{t("alertsUi.bellTitle")}</DropdownMenuLabel>
        <DropdownMenuSeparator />
        <AlertsList />
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function AlertsList() {
  const { t } = useTranslation();
  const { data: alerts, isLoading } = useAlerts(false);
  const ack = useAcknowledgeAlert();
  const navigate = useNavigate();

  if (isLoading) {
    return (
      <div className="flex justify-center p-3">
        <Loader2 className="h-4 w-4 animate-spin text-primary" />
      </div>
    );
  }

  if (!alerts || alerts.length === 0) {
    return (
      <div className="flex flex-col items-center gap-2 p-4 text-center">
        <AlertTriangle className="h-6 w-6 text-muted-foreground" />
        <span className="text-xs text-muted-foreground">{t("alertsUi.empty")}</span>
      </div>
    );
  }

  const onClickField = (alert: AlertRead) => {
    navigate("/fields");
    void ack.mutateAsync(alert.id);
  };

  return (
    <div className="max-h-[420px] overflow-y-auto">
      {alerts.map((alert) => {
        const Icon = TYPE_ICONS[alert.type] ?? AlertTriangle;
        return (
          <div
            key={alert.id}
            className="cursor-pointer border-b px-2 py-2 last:border-b-0 hover:bg-accent/40"
            onClick={() => onClickField(alert)}
          >
            <div className="flex items-start gap-2">
              <Icon className={cn("mt-0.5 h-4 w-4 flex-shrink-0", SEVERITY_COLORS[alert.severity])} />
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <Badge
                    variant={alert.severity === "critical" ? "destructive" : "secondary"}
                    className="px-1.5 text-[10px] font-normal"
                  >
                    {t(`alertsUi.severity.${alert.severity}`)}
                  </Badge>
                  <span className="text-[10px] text-muted-foreground">
                    {format(new Date(alert.created_at), "d MMM, HH:mm", { locale: uk })}
                  </span>
                </div>
                <div className="mt-0.5 text-xs leading-tight">{alert.message_uk}</div>
              </div>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6 flex-shrink-0"
                onClick={(e) => {
                  e.stopPropagation();
                  void ack.mutateAsync(alert.id);
                }}
                title={t("alertsUi.ackAll")}
              >
                <Check className="h-3.5 w-3.5" />
              </Button>
            </div>
          </div>
        );
      })}
    </div>
  );
}
