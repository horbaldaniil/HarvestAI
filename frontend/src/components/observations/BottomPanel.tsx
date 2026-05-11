import { useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import {
  ChevronDown,
  ChevronUp,
  Download,
  Loader2,
  RefreshCw,
  Satellite,
} from "lucide-react";
import type { AxiosError } from "axios";

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import type { FieldRead } from "@/api/fields";
import type { IndexName } from "@/api/observations";
import { downloadFieldReport, triggerBrowserDownload } from "@/api/reports";
import {
  useObservations,
  useRefreshObservations,
  useRequestHeatmap,
  useTrackObservationJob,
} from "@/hooks/useObservations";
import { IndexChart } from "./IndexChart";
import { PredictionCard } from "./PredictionCard";

const INDICES: IndexName[] = ["ndvi", "evi", "ndwi", "savi"];

interface BottomPanelProps {
  field: FieldRead;
  selectedDate: string | null;
  activeIndex: IndexName;
  onSelectDate: (date: string | null) => void;
  onIndexChange: (index: IndexName) => void;
  /**
   * Optional RQ job id that was kicked off elsewhere (e.g. by the field
   * create endpoint). If set, the panel subscribes to its SSE stream so
   * the user sees auto-fetch progress without clicking Refresh manually.
   */
  pendingJobId?: string | null;
}

export function BottomPanel({
  field,
  selectedDate,
  activeIndex,
  onSelectDate,
  onIndexChange,
  pendingJobId,
}: BottomPanelProps) {
  const { t } = useTranslation();
  const [collapsed, setCollapsed] = useState(false);
  const [exporting, setExporting] = useState(false);

  const obsQuery = useObservations(field.id);
  const refreshMut = useRefreshObservations();
  const heatmapMut = useRequestHeatmap();
  // Subscribe to the auto-fetch SSE if the parent passed a job id.
  const trackedProgress = useTrackObservationJob(pendingJobId, field.id);

  const observations = obsQuery.data ?? [];
  // Either the manual refresh or the tracked auto-fetch may be active.
  const activeProgress = refreshMut.progress ?? trackedProgress;
  const isRefreshing =
    refreshMut.isPending ||
    (activeProgress &&
      ["queued", "running", "connected"].includes(activeProgress.state));
  const progressPct = activeProgress?.progress
    ? Math.round(activeProgress.progress * 100)
    : 0;

  const handleRefresh = async () => {
    try {
      await refreshMut.mutateAsync(field.id);
    } catch (err) {
      const ax = err as AxiosError<{ detail?: string }>;
      toast.error(ax.response?.data?.detail ?? t("observations.refreshFailed"));
    }
  };

  const handleExport = async () => {
    setExporting(true);
    try {
      const blob = await downloadFieldReport(field.id);
      const safe = field.name.replace(/[^\p{L}\p{N}_-]+/gu, "_").slice(0, 60);
      triggerBrowserDownload(blob, `HarvestAI_${safe}_${field.season_year}.pdf`);
      toast.success(t("report.success"));
    } catch {
      toast.error(t("report.error"));
    } finally {
      setExporting(false);
    }
  };

  const handlePointClick = (date: string) => {
    onSelectDate(date);
    heatmapMut.mutate(
      { fieldId: field.id, date, index: activeIndex },
      {
        onError: (err) => {
          const ax = err as AxiosError<{ detail?: string }>;
          toast.error(ax.response?.data?.detail ?? t("observations.heatmap.failed"));
        },
      },
    );
  };

  return (
    <div
      className={cn(
        "pointer-events-auto absolute bottom-0 left-0 right-0 z-[900] border-t bg-card shadow-2xl transition-all duration-300",
        collapsed ? "h-12" : "h-[340px]",
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between border-b px-4 py-2">
        <div className="flex items-center gap-3">
          <Satellite className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold">{field.name}</h3>
          <span className="text-xs text-muted-foreground">
            {observations.length} {t("fields.area").toLowerCase()}
          </span>
        </div>

        <div className="flex items-center gap-2">
          {/* Index tabs */}
          {!collapsed && (
            <div className="flex gap-1 rounded-md border p-0.5">
              {INDICES.map((idx) => (
                <button
                  key={idx}
                  onClick={() => onIndexChange(idx)}
                  className={cn(
                    "rounded px-2 py-1 text-xs font-medium transition-colors",
                    activeIndex === idx
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:bg-accent",
                  )}
                  title={t(`indices.descriptions.${idx}`)}
                >
                  {t(`indices.${idx}`)}
                </button>
              ))}
            </div>
          )}

          <Button
            variant="ghost"
            size="sm"
            onClick={handleRefresh}
            disabled={!!isRefreshing}
          >
            {isRefreshing ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <RefreshCw className="h-3.5 w-3.5" />
            )}
            {isRefreshing ? t("observations.refreshing") : t("observations.refresh")}
          </Button>

          <Button
            variant="ghost"
            size="sm"
            onClick={handleExport}
            disabled={exporting}
            title={t("report.export")}
          >
            {exporting ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Download className="h-3.5 w-3.5" />
            )}
            {exporting ? t("report.exporting") : t("report.export")}
          </Button>

          <Button
            variant="ghost"
            size="icon"
            onClick={() => setCollapsed((c) => !c)}
            className="h-7 w-7"
            title={collapsed ? "Розгорнути" : "Згорнути"}
          >
            {collapsed ? (
              <ChevronUp className="h-4 w-4" />
            ) : (
              <ChevronDown className="h-4 w-4" />
            )}
          </Button>
        </div>
      </div>

      {/* Progress bar during refresh */}
      {isRefreshing && (
        <div className="px-4 pt-2">
          <Progress value={progressPct} />
          <div className="mt-1 text-center text-xs text-muted-foreground">
            {t(`jobs.${activeProgress?.state ?? "queued"}` as const)}
            {progressPct > 0 ? ` — ${progressPct}%` : ""}
          </div>
        </div>
      )}

      {/* Chart body — split layout: chart on the left, PredictionCard on the right */}
      {!collapsed && (
        <div className="flex h-[calc(100%-3rem)]">
          <div className="relative flex-1 px-2 pb-2 pt-1">
            {obsQuery.isLoading ? (
              <div className="flex h-full items-center justify-center">
                <Loader2 className="h-5 w-5 animate-spin text-primary" />
              </div>
            ) : observations.length === 0 ? (
              <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
                <p className="text-sm font-semibold">{t("observations.empty.title")}</p>
                <p className="max-w-sm text-xs text-muted-foreground">
                  {t("observations.empty.description")}
                </p>
                <Button size="sm" onClick={handleRefresh} disabled={!!isRefreshing}>
                  <RefreshCw className="h-3.5 w-3.5" />
                  {t("observations.empty.cta")}
                </Button>
              </div>
            ) : (
              <>
                <div className="absolute right-6 top-2 text-[10px] text-muted-foreground">
                  {t("observations.clickHint")}
                </div>
                <IndexChart
                  observations={observations}
                  index={activeIndex}
                  selectedDate={selectedDate}
                  onPointClick={handlePointClick}
                />
              </>
            )}
          </div>
          <div className="w-72 flex-shrink-0 border-l">
            <PredictionCard fieldId={field.id} />
          </div>
        </div>
      )}
    </div>
  );
}
