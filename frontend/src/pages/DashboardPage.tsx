import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";
import { format } from "date-fns";
import { uk } from "date-fns/locale";
import {
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  Bell,
  Leaf,
  Loader2,
  Minus,
  Sprout,
  TrendingUp,
} from "lucide-react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { AppShell } from "@/components/layout/AppShell";
import { useDashboard } from "@/hooks/useDashboard";
import { useAlerts, useAcknowledgeAlert } from "@/hooks/useAlerts";
import { CROP_COLORS } from "@/lib/colors";
import { cn } from "@/lib/utils";
import type { DashboardFieldRow, YearOverYear as YoYType } from "@/api/dashboard";

export function DashboardPage() {
  const { t } = useTranslation();
  const { data, isLoading } = useDashboard();

  if (isLoading || !data) {
    return (
      <AppShell>
        <div className="flex h-64 items-center justify-center">
          <Loader2 className="h-6 w-6 animate-spin text-primary" />
        </div>
      </AppShell>
    );
  }

  const { kpis, fields, yoy } = data;

  return (
    <AppShell>
      <div className="space-y-6">
        <header>
          <h1 className="text-3xl font-bold tracking-tight">{t("dashboard.title")}</h1>
          <p className="text-muted-foreground">{t("app.tagline")}</p>
        </header>

        {/* KPI cards */}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <KpiCard
            label={t("dashboardKpi.totalFields")}
            value={kpis.total_fields.toString()}
            icon={<Sprout className="h-5 w-5 text-harvest-600" />}
          />
          <KpiCard
            label={t("dashboardKpi.totalArea")}
            value={kpis.total_area_ha.toFixed(1)}
            icon={<Leaf className="h-5 w-5 text-harvest-700" />}
          />
          <KpiCard
            label={t("dashboardKpi.predictedYield")}
            value={
              kpis.predicted_total_yield_t !== null
                ? `${kpis.predicted_total_yield_t.toFixed(1)} т`
                : t("dashboardKpi.noData")
            }
            icon={<TrendingUp className="h-5 w-5 text-emerald-600" />}
          />
          <KpiCard
            label={t("dashboardKpi.activeAlerts")}
            value={kpis.active_alerts_count.toString()}
            icon={
              <Bell
                className={cn(
                  "h-5 w-5",
                  kpis.active_alerts_count > 0
                    ? "text-amber-600"
                    : "text-muted-foreground",
                )}
              />
            }
            highlight={kpis.active_alerts_count > 0}
          />
        </div>

        {/* Fields table + Year-over-Year side-by-side */}
        <div className="grid gap-4 lg:grid-cols-3">
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle className="text-base">{t("dashboardTable.title")}</CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <FieldsTable fields={fields} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">{t("dashboardYoY.title")}</CardTitle>
              <CardDescription>
                {yoy.current_year}
                {yoy.prev_year_avg_ndvi !== null
                  ? ` vs ${yoy.current_year - 1}`
                  : ""}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <YearOverYearWidget yoy={yoy} />
            </CardContent>
          </Card>
        </div>

        {/* Recent alerts */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">{t("alertsUi.bellTitle")}</CardTitle>
          </CardHeader>
          <CardContent>
            <RecentAlertsList />
          </CardContent>
        </Card>
      </div>
    </AppShell>
  );
}

function KpiCard({
  label,
  value,
  icon,
  highlight = false,
}: {
  label: string;
  value: string;
  icon: React.ReactNode;
  highlight?: boolean;
}) {
  return (
    <Card className={highlight ? "border-amber-300" : ""}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">
          {label}
        </CardTitle>
        {icon}
      </CardHeader>
      <CardContent>
        <div className="text-3xl font-bold tabular-nums">{value}</div>
      </CardContent>
    </Card>
  );
}

function FieldsTable({ fields }: { fields: DashboardFieldRow[] }) {
  const { t } = useTranslation();
  const navigate = useNavigate();
  if (!fields || fields.length === 0) {
    return (
      <div className="p-6 text-center text-sm text-muted-foreground">
        {t("dashboardTable.empty")}
      </div>
    );
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b text-left text-xs text-muted-foreground">
            <th className="px-4 py-2 font-medium">{t("dashboardTable.name")}</th>
            <th className="px-2 py-2 font-medium">{t("dashboardTable.crop")}</th>
            <th className="px-2 py-2 text-right font-medium">
              {t("dashboardTable.area")}
            </th>
            <th className="px-2 py-2 text-right font-medium">
              {t("dashboardTable.ndvi")}
            </th>
            <th className="px-2 py-2 text-right font-medium">
              {t("dashboardTable.predicted")}
            </th>
            <th className="px-2 py-2 font-medium">{t("dashboardTable.status")}</th>
          </tr>
        </thead>
        <tbody>
          {fields.map((f) => (
            <tr
              key={f.field_id}
              className="cursor-pointer border-b last:border-b-0 hover:bg-accent/40"
              onClick={() => navigate(`/fields?selected=${f.field_id}`)}
            >
              <td className="px-4 py-2 font-medium">{f.name}</td>
              <td className="px-2 py-2">
                <div className="flex items-center gap-1.5">
                  <span
                    className="inline-block h-2.5 w-2.5 rounded-sm"
                    style={{ background: CROP_COLORS[f.crop_type] }}
                  />
                  <span>{t(`fields.crops.${f.crop_type}`)}</span>
                </div>
              </td>
              <td className="px-2 py-2 text-right tabular-nums">
                {f.area_ha.toFixed(1)}
              </td>
              <td className="px-2 py-2 text-right tabular-nums">
                {f.current_ndvi !== null ? f.current_ndvi.toFixed(2) : "—"}
              </td>
              <td className="px-2 py-2 text-right tabular-nums">
                {f.predicted_tha !== null ? `${f.predicted_tha.toFixed(1)} т/га` : "—"}
              </td>
              <td className="px-2 py-2">
                {f.has_alerts ? (
                  <Badge variant="destructive" className="font-normal">
                    {t("dashboardTable.alert")}
                  </Badge>
                ) : (
                  <Badge variant="secondary" className="font-normal">
                    {t("dashboardTable.ok")}
                  </Badge>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function YearOverYearWidget({ yoy }: { yoy: YoYType }) {
  const { t } = useTranslation();
  const hasPrev = yoy.prev_year_avg_ndvi !== null;
  return (
    <div className="space-y-3">
      <div className="flex items-baseline justify-between">
        <span className="text-xs text-muted-foreground">{t("dashboardYoY.current")}</span>
        <span className="text-2xl font-bold tabular-nums">
          {yoy.current_year_avg_ndvi !== null
            ? yoy.current_year_avg_ndvi.toFixed(2)
            : "—"}
        </span>
      </div>
      {hasPrev ? (
        <>
          <div className="flex items-baseline justify-between">
            <span className="text-xs text-muted-foreground">{t("dashboardYoY.previous")}</span>
            <span className="text-lg tabular-nums text-muted-foreground">
              {(yoy.prev_year_avg_ndvi ?? 0).toFixed(2)}
            </span>
          </div>
          <div className="flex items-center justify-between border-t pt-2">
            <span className="text-xs text-muted-foreground">{t("dashboardYoY.diff")}</span>
            <DiffIndicator pct={yoy.diff_pct ?? 0} />
          </div>
        </>
      ) : (
        <p className="text-xs text-muted-foreground">{t("dashboardYoY.noPrevYear")}</p>
      )}
    </div>
  );
}

function DiffIndicator({ pct }: { pct: number }) {
  if (Math.abs(pct) < 0.5) {
    return (
      <div className="flex items-center gap-1 text-muted-foreground">
        <Minus className="h-4 w-4" />
        <span className="tabular-nums">0.0%</span>
      </div>
    );
  }
  const positive = pct > 0;
  const Icon = positive ? ArrowUpRight : ArrowDownRight;
  const color = positive ? "text-emerald-600" : "text-red-600";
  return (
    <div className={cn("flex items-center gap-1 font-medium", color)}>
      <Icon className="h-4 w-4" />
      <span className="tabular-nums">
        {positive ? "+" : ""}
        {pct.toFixed(1)}%
      </span>
    </div>
  );
}

function RecentAlertsList() {
  const { t } = useTranslation();
  const { data: alerts, isLoading } = useAlerts(false);
  const ack = useAcknowledgeAlert();

  if (isLoading) {
    return (
      <div className="flex justify-center py-6">
        <Loader2 className="h-5 w-5 animate-spin text-primary" />
      </div>
    );
  }
  if (!alerts || alerts.length === 0) {
    return (
      <div className="flex flex-col items-center gap-2 py-6">
        <AlertTriangle className="h-6 w-6 text-muted-foreground" />
        <span className="text-xs text-muted-foreground">{t("alertsUi.empty")}</span>
      </div>
    );
  }
  return (
    <div className="space-y-2">
      {alerts.slice(0, 5).map((a) => (
        <div
          key={a.id}
          className="flex items-start gap-2 rounded-md border p-2 text-sm"
        >
          <Badge
            variant={a.severity === "critical" ? "destructive" : "secondary"}
            className="text-[10px] font-normal"
          >
            {t(`alertsUi.severity.${a.severity}`)}
          </Badge>
          <div className="min-w-0 flex-1">
            <div className="flex items-baseline gap-2">
              {a.field_name && (
                <span className="truncate text-xs font-medium">{a.field_name}</span>
              )}
              <span className="text-[10px] text-muted-foreground">
                {format(new Date(a.created_at), "d MMM HH:mm", { locale: uk })}
              </span>
            </div>
            <div className="mt-0.5 text-xs leading-tight">{a.message_uk}</div>
          </div>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => void ack.mutateAsync(a.id)}
            className="h-7 text-xs"
          >
            ✓
          </Button>
        </div>
      ))}
    </div>
  );
}
