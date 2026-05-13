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

  // Render the q05/q95 band only when both bounds are present. XGBoost
  // payloads carry quantile siblings; stack / RF / LGBM don't, so this
  // row simply disappears for those families.
  const hasRange =
    prediction.value_tha_q05 !== null && prediction.value_tha_q95 !== null;

  // Resolve the model family from `model_name` ("yield_<family>_<crop>")
  // so we can suppress the disclaimer when the explainer matches the
  // headline model — e.g. for a plain XGBoost prediction, the SHAP
  // already comes from XGBoost and there's nothing to disclose.
  const modelFamily = prediction.model_name.split("_")[1] ?? null;
  const showExplainerDisclaimer =
    prediction.explainer_source !== null &&
    prediction.explainer_source !== modelFamily;

  return (
    // Three-band layout: fixed header (number + range) on top, scrollable
    // body (LLM summary + SHAP bars) in the middle, fixed footer (progress
    // bar + Перерахувати button) at the bottom. The middle band is
    // `flex-1 min-h-0 overflow-y-auto` — `min-h-0` is the magic that lets
    // a flex item shrink below its content height and scroll internally;
    // without it Tailwind's default `min-h: auto` would keep the
    // summary block at its natural height and push the button off the
    // bottom of the 292 px-tall right column when the LLM produces a
    // long narrative.
    <div className="flex h-full flex-col p-3">
      {/* Big number — fixed */}
      <div className="shrink-0 text-center">
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
        {hasRange && (
          <div className="mt-0.5 text-[11px] tabular-nums text-muted-foreground">
            ↕ {prediction.value_tha_q05!.toFixed(1)} —{" "}
            {prediction.value_tha_q95!.toFixed(1)} т/га
            <span className="ml-1 text-muted-foreground/70">
              (90% довірчий інтервал)
            </span>
          </div>
        )}
      </div>

      {/* Scrollable middle band — summary + SHAP bars share this area. */}
      <div className="mt-3 flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto pr-1">
        {/* LLM narrative — italic, 2-3 sentences explaining the value. */}
        {prediction.summary_text && (
          <p className="shrink-0 rounded-md border bg-muted/40 p-2 text-xs italic leading-relaxed text-muted-foreground">
            {prediction.summary_text}
          </p>
        )}

        {/* SHAP top-5 */}
        {prediction.shap_top_json.length > 0 && (
          <div className="shrink-0">
            <div className="mb-1 text-xs font-semibold text-muted-foreground">
              {t("prediction.factors")}
            </div>
            <div className="space-y-1">
              {prediction.shap_top_json.map((bar) => (
                <ShapBarRow key={bar.name} bar={bar} />
              ))}
            </div>
            {showExplainerDisclaimer && (
              <p className="mt-2 text-[10px] text-muted-foreground/70">
                Пояснення на основі моделі {prediction.explainer_source}{" "}
                (компонент стек-ансамблю)
              </p>
            )}
          </div>
        )}
      </div>

      {/* Progress while recomputing — fixed footer above the button. */}
      {isWorking && (
        <div className="mt-3 shrink-0">
          <Progress value={progressPct} />
        </div>
      )}

      <Button
        size="sm"
        variant="outline"
        onClick={onRecompute}
        disabled={!!isWorking}
        className="mt-3 w-full shrink-0"
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
