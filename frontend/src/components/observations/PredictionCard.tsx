import { useTranslation } from "react-i18next";
import { Loader2, RefreshCw, Sparkles } from "lucide-react";
import type { AxiosError } from "axios";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useLatestPrediction, useRecomputePrediction } from "@/hooks/usePrediction";
import type { ShapBar } from "@/api/predictions";

interface PredictionCardProps {
  fieldId: number;
}

export function PredictionCard({ fieldId }: PredictionCardProps) {
  const { t } = useTranslation();
  const { data: prediction, isLoading } = useLatestPrediction(fieldId);
  const recompute = useRecomputePrediction();

  const isWorking =
    recompute.isPending ||
    (recompute.progress &&
      ["queued", "running", "connected"].includes(recompute.progress.state));
  const progressPct = recompute.progress?.progress
    ? Math.round(recompute.progress.progress * 100)
    : 0;

  const onRecompute = async () => {
    try {
      await recompute.mutateAsync(fieldId);
    } catch (err) {
      const ax = err as AxiosError<{ detail?: string }>;
      toast.error(ax.response?.data?.detail ?? "Не вдалося оновити");
    }
  };

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-5 w-5 animate-spin text-primary" />
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 p-3 text-center">
        <Sparkles className="h-8 w-8 text-muted-foreground" />
        <div>
          <div className="text-sm font-semibold">{t("prediction.noData")}</div>
          <div className="mt-1 text-xs text-muted-foreground">
            {t("prediction.noDataHint")}
          </div>
        </div>
        <Button size="sm" onClick={onRecompute} disabled={!!isWorking}>
          {isWorking ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <RefreshCw className="h-3.5 w-3.5" />
          )}
          {t("prediction.retrain")}
        </Button>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col gap-3 p-3">
      {/* Big number */}
      <div className="text-center">
        <div className="text-3xl font-bold tabular-nums">
          {prediction.value_tha.toFixed(1)}
          {prediction.confidence !== null && (
            <span className="ml-1 text-sm text-muted-foreground">
              ± {prediction.confidence.toFixed(1)}
            </span>
          )}
        </div>
        <div className="text-xs text-muted-foreground">
          {t("prediction.unit")} · {t("prediction.modelLabel")}:{" "}
          {prediction.model_version}
        </div>
      </div>

      {/* SHAP top-5 */}
      {prediction.shap_top_json.length > 0 && (
        <div className="flex-1 overflow-y-auto">
          <div className="mb-1 text-xs font-semibold text-muted-foreground">
            {t("prediction.factors")}
          </div>
          <div className="space-y-1">
            {prediction.shap_top_json.map((bar) => (
              <ShapBarRow key={bar.name} bar={bar} />
            ))}
          </div>
        </div>
      )}

      {/* Progress while recomputing */}
      {isWorking && (
        <div>
          <Progress value={progressPct} />
        </div>
      )}

      <Button
        size="sm"
        variant="outline"
        onClick={onRecompute}
        disabled={!!isWorking}
        className="w-full"
      >
        {isWorking ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
        ) : (
          <RefreshCw className="h-3.5 w-3.5" />
        )}
        {t("prediction.retrain")}
      </Button>
    </div>
  );
}

function ShapBarRow({ bar }: { bar: ShapBar }) {
  const { t } = useTranslation();
  const positive = bar.contribution >= 0;
  // Bar width relative to max likely abs contribution.
  const widthPct = Math.min(100, Math.abs(bar.contribution) * 80);
  const featureLabel = t(`prediction.featureNames.${bar.name}` as const, {
    defaultValue: bar.name,
  });

  return (
    <div className="text-xs">
      <div className="flex items-baseline justify-between">
        <span className="truncate" title={featureLabel}>
          {featureLabel}
        </span>
        <span
          className={
            "ml-2 flex-shrink-0 tabular-nums " +
            (positive ? "text-emerald-600" : "text-red-600")
          }
        >
          {positive ? "+" : ""}
          {bar.contribution.toFixed(2)}
        </span>
      </div>
      <div className="mt-0.5 h-1.5 w-full rounded-full bg-muted">
        <div
          className={
            "h-full rounded-full " +
            (positive ? "bg-emerald-500" : "bg-red-500")
          }
          style={{ width: `${widthPct}%` }}
        />
      </div>
    </div>
  );
}
