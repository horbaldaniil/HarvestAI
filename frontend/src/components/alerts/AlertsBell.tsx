import { useState } from "react";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";
import { format } from "date-fns";
import { uk } from "date-fns/locale";
import {
  AlertTriangle,
  Bell,
  Droplets,
  Eye,
  EyeOff,
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
        <Button
          variant="ghost"
          size="icon"
          className="relative h-9 w-9"
          aria-label={t("alertsUi.bellTitle")}
        >
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
      <DropdownMenuContent align="end" className="w-[22rem]">
        <AlertsList />
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function AlertsList() {
  const { t } = useTranslation();
  const [showAcknowledged, setShowAcknowledged] = useState(false);
  const { data: alerts, isLoading } = useAlerts(showAcknowledged);
  const ack = useAcknowledgeAlert();
  const navigate = useNavigate();

  return (
    <>
      <div className="flex items-center justify-between px-2 py-1.5">
        <DropdownMenuLabel className="px-0 py-0">
          {t("alertsUi.bellTitle")}
        </DropdownMenuLabel>
        <Button
          variant="ghost"
          size="sm"
          className="h-6 px-2 text-[10px]"
          onClick={() => setShowAcknowledged((s) => !s)}
          title={
            showAcknowledged
              ? "Показати тільки непрочитані"
              : "Показати прочитані"
          }
        >
          {showAcknowledged ? (
            <>
              <EyeOff className="h-3 w-3" />
              Непрочитані
            </>
          ) : (
            <>
              <Eye className="h-3 w-3" />
              Прочитані
            </>
          )}
        </Button>
      </div>
      <DropdownMenuSeparator />

      {isLoading ? (
        <div className="flex justify-center p-3">
          <Loader2 className="h-4 w-4 animate-spin text-primary" />
        </div>
      ) : !alerts || alerts.length === 0 ? (
        <div className="flex flex-col items-center gap-2 p-4 text-center">
          <AlertTriangle className="h-6 w-6 text-muted-foreground" />
          <span className="text-xs text-muted-foreground">
            {showAcknowledged ? "Прочитаних сповіщень немає" : t("alertsUi.empty")}
          </span>
        </div>
      ) : (
        <div className="max-h-[420px] overflow-y-auto">
          {alerts.map((alert) => (
            <AlertRow
              key={alert.id}
              alert={alert}
              onOpenField={() => {
                if (alert.field_id) {
                  navigate(`/fields?selected=${alert.field_id}`);
                }
              }}
              onMarkRead={() => void ack.mutateAsync(alert.id)}
            />
          ))}
        </div>
      )}
    </>
  );
}

function AlertRow({
  alert,
  onOpenField,
  onMarkRead,
}: {
  alert: AlertRead;
  onOpenField: () => void;
  onMarkRead: () => void;
}) {
  const { t } = useTranslation();
  const Icon = TYPE_ICONS[alert.type] ?? AlertTriangle;
  return (
    <div
      className={cn(
        "cursor-pointer border-b px-2 py-2 last:border-b-0 hover:bg-accent/40",
        alert.acknowledged && "opacity-60",
      )}
      onClick={onOpenField}
    >
      <div className="flex items-start gap-2">
        <Icon
          className={cn(
            "mt-0.5 h-4 w-4 flex-shrink-0",
            SEVERITY_COLORS[alert.severity],
          )}
        />
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
          {alert.field_name && (
            <div className="mt-0.5 text-[10px] text-muted-foreground">
              Поле: <span className="font-medium">{alert.field_name}</span>
            </div>
          )}
        </div>
        {!alert.acknowledged && (
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 flex-shrink-0"
            onClick={(e) => {
              e.stopPropagation();
              onMarkRead();
            }}
            title="Позначити прочитаним"
          >
            <Eye className="h-3.5 w-3.5" />
          </Button>
        )}
      </div>
    </div>
  );
}
