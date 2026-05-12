import { useState } from "react";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";
import { format } from "date-fns";
import { uk } from "date-fns/locale";
import { toast } from "sonner";
import {
  AlertTriangle,
  Bell,
  Download,
  Droplets,
  Eye,
  EyeOff,
  Leaf,
  Loader2,
  Sprout,
  TrendingUp,
} from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { AppShell } from "@/components/layout/AppShell";
import { CropCalendarTimeline } from "@/components/dashboard/CropCalendarTimeline";
import { DashboardMap } from "@/components/dashboard/DashboardMap";
import { FilterBar } from "@/components/dashboard/FilterBar";
import { IncomeProjectionCard } from "@/components/dashboard/IncomeProjectionCard";
import { OblastComparisonTable } from "@/components/dashboard/OblastComparisonTable";
import { TopMoversCard } from "@/components/dashboard/TopMoversCard";
import { WeatherSummaryCard } from "@/components/dashboard/WeatherSummaryCard";
import { useDashboard } from "@/hooks/useDashboard";
import { useAlerts, useAcknowledgeAlert } from "@/hooks/useAlerts";
import { downloadPortfolioReport, triggerBrowserDownload } from "@/api/reports";
import { CROP_COLORS } from "@/lib/colors";
import { cn } from "@/lib/utils";
import type { CropType, DashboardFieldRow } from "@/api/dashboard";

export function DashboardPage() {
  const { t } = useTranslation();
  const [selectedCrops, setSelectedCrops] = useState<CropType[]>([]);
  const { data, isLoading } = useDashboard({ crops: selectedCrops });
  const [exporting, setExporting] = useState(false);

  if (isLoading || !data) {
    return (
      <AppShell>
        <div className="flex h-64 items-center justify-center">
          <Loader2 className="h-6 w-6 animate-spin text-primary" />
        </div>
      </AppShell>
    );
  }

  const { kpis, fields, top_movers, weather_by_field, yoy } = data;

  const handleExport = async () => {
    setExporting(true);
    try {
      const blob = await downloadPortfolioReport();
      const stamp = new Date().toISOString().slice(0, 16).replace(/[:T]/g, "");
      triggerBrowserDownload(blob, `HarvestAI_portfolio_${stamp}.pdf`);
      toast.success(t("report.success"));
    } catch {
      toast.error(t("report.error"));
    } finally {
      setExporting(false);
    }
  };

  return (
    <AppShell>
      <div className="space-y-6">
        <header className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">
              {t("dashboard.title")}
            </h1>
            <p className="text-muted-foreground">{t("app.tagline")}</p>
          </div>
          <Button onClick={handleExport} disabled={exporting} variant="outline">
            {exporting ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Download className="h-4 w-4" />
            )}
            {exporting ? t("report.exporting") : t("report.portfolioExport")}
          </Button>
        </header>

        <FilterBar
          selectedCrops={selectedCrops}
          onCropsChange={setSelectedCrops}
        />

        {/* Hero mini-map: every field polygon coloured by latest NDVI */}
        <DashboardMap fields={fields} />

        {/* KPI cards */}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
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
            label="Середній NDWI"
            value={
              kpis.avg_ndwi_current !== null
                ? kpis.avg_ndwi_current.toFixed(2)
                : "—"
            }
            icon={<Droplets className="h-5 w-5 text-blue-600" />}
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

        {/* Income projection */}
        <IncomeProjectionCard fields={fields} />

        {/* Fields table + Top movers + Weather */}
        <div className="grid gap-4 lg:grid-cols-3">
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle className="text-base">{t("dashboardTable.title")}</CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <FieldsTable fields={fields} />
            </CardContent>
          </Card>

          <div className="space-y-4">
            <TopMoversCard movers={top_movers} currentYear={yoy.current_year} />
            <WeatherSummaryCard weatherByField={weather_by_field} />
          </div>
        </div>

        {/* Oblast comparison + Crop calendar */}
        <div className="grid gap-4 lg:grid-cols-2">
          <OblastComparisonTable fields={fields} />
          <CropCalendarTimeline breakdown={data.crops_breakdown} />
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
            <th className="px-2 py-2 text-right font-medium">Risk</th>
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
              <td className="px-2 py-2 text-right tabular-nums">
                <RiskBadge score={f.risk_score} factors={f.risk_factors} />
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

function RiskBadge({ score, factors }: { score: number; factors: string[] }) {
  const tier =
    score >= 50 ? "critical" : score >= 25 ? "warning" : "ok";
  const colour = {
    critical: "text-red-700 bg-red-50 border-red-200",
    warning: "text-amber-700 bg-amber-50 border-amber-200",
    ok: "text-emerald-700 bg-emerald-50 border-emerald-200",
  }[tier];
  const tierLabel = {
    critical: "Високий ризик",
    warning: "Помірний ризик",
    ok: "Низький ризик",
  }[tier];
  const tierEmoji = { critical: "🔴", warning: "🟡", ok: "🟢" }[tier];

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          className={cn(
            "inline-block cursor-help rounded border px-1.5 py-0.5 text-xs tabular-nums",
            colour,
          )}
          // onClick stops table-row navigation when the user just wants the tooltip.
          onClick={(e) => e.stopPropagation()}
        >
          {score}
        </span>
      </TooltipTrigger>
      <TooltipContent side="top" align="end" className="w-64">
        <div className="space-y-1.5">
          <div className="flex items-center justify-between font-semibold">
            <span>
              {tierEmoji} {tierLabel}
            </span>
            <span className="tabular-nums text-muted-foreground">
              {score} / 100
            </span>
          </div>
          {factors.length > 0 ? (
            <ul className="space-y-1 border-t pt-1.5 text-[11px] leading-tight">
              {factors.map((f, i) => (
                <li key={i} className="flex items-start gap-1.5">
                  <span className="mt-0.5 text-amber-500">•</span>
                  <span>{f}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="border-t pt-1.5 text-[11px] text-muted-foreground">
              Без помітних ризиків — поле в нормі.
            </p>
          )}
        </div>
      </TooltipContent>
    </Tooltip>
  );
}

function RecentAlertsList() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [showAcknowledged, setShowAcknowledged] = useState(false);
  const { data: alerts, isLoading } = useAlerts(showAcknowledged);
  const ack = useAcknowledgeAlert();

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-end pb-1">
        <Button
          variant="ghost"
          size="sm"
          className="h-7 px-2 text-xs"
          onClick={() => setShowAcknowledged((s) => !s)}
        >
          {showAcknowledged ? (
            <>
              <EyeOff className="h-3.5 w-3.5" />
              Непрочитані
            </>
          ) : (
            <>
              <Eye className="h-3.5 w-3.5" />
              Показати прочитані
            </>
          )}
        </Button>
      </div>

      {isLoading ? (
        <div className="flex justify-center py-6">
          <Loader2 className="h-5 w-5 animate-spin text-primary" />
        </div>
      ) : !alerts || alerts.length === 0 ? (
        <div className="flex flex-col items-center gap-2 py-6">
          <AlertTriangle className="h-6 w-6 text-muted-foreground" />
          <span className="text-xs text-muted-foreground">
            {showAcknowledged
              ? "Прочитаних сповіщень немає"
              : t("alertsUi.empty")}
          </span>
        </div>
      ) : (
        alerts.slice(0, 10).map((a) => (
          <div
            key={a.id}
            className={cn(
              "flex cursor-pointer items-start gap-2 rounded-md border p-2 text-sm hover:bg-accent/30",
              a.acknowledged && "opacity-60",
            )}
            onClick={() => {
              if (a.field_id) navigate(`/fields?selected=${a.field_id}`);
            }}
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
                  <span className="truncate text-xs font-medium">
                    {a.field_name}
                  </span>
                )}
                <span className="text-[10px] text-muted-foreground">
                  {format(new Date(a.created_at), "d MMM HH:mm", { locale: uk })}
                </span>
              </div>
              <div className="mt-0.5 text-xs leading-tight">{a.message_uk}</div>
            </div>
            {!a.acknowledged && (
              <Button
                size="icon"
                variant="ghost"
                onClick={(e) => {
                  e.stopPropagation();
                  void ack.mutateAsync(a.id);
                }}
                className="h-7 w-7 flex-shrink-0"
                title="Позначити прочитаним"
              >
                <Eye className="h-3.5 w-3.5" />
              </Button>
            )}
          </div>
        ))
      )}
    </div>
  );
}
