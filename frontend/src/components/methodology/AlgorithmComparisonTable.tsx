import { useMemo, useState } from "react";
import { Loader2 } from "lucide-react";

import type { LeaderboardRow, ModelInfo } from "@/api/methodology";
import { useLeaderboard } from "@/hooks/useMethodology";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

/**
 * Per-crop drill-down of all 6 model families with test metrics +
 * CV-variance sparkline. Replaces the legacy v2 3×3 hardcoded grid
 * (wheat/corn/sunflower × rf/xgboost/lstm).
 *
 * Layout: crop selector top, then 6 rows (one per family) — for the
 * selected crop. Cells show test R², MAE, MAPE plus an inline
 * sparkline of the RepeatedKFold(5×3) variance (mean ± σ). Stack
 * rows are highlighted as the recommended default.
 *
 * Backward compat: still accepts the legacy `models` prop (ignored)
 * so the old `<AlgorithmComparisonTable models={...} />` callsite
 * doesn't break — we now read everything from `useLeaderboard()`.
 */
interface Props {
  /** Legacy prop from v2 wiring — ignored. Kept for back-compat. */
  models?: ModelInfo[];
}

const CROP_LABEL_UK: Record<string, string> = {
  wheat: "Пшениця",
  corn: "Кукурудза",
  sunflower: "Соняшник",
  soybean: "Соя",
  rapeseed: "Ріпак",
  barley: "Ячмінь",
  rye: "Жито",
  oats: "Овес",
  buckwheat: "Гречка",
  peas: "Горох",
  sugar_beet: "Цукровий буряк",
  potato: "Картопля",
  corn_silage: "Кукурудза на силос",
};

const FAMILY_LABEL: Record<string, string> = {
  rf: "Random Forest",
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
  lstm: "LSTM",
  stack: "Stacked Ensemble",
};

const FAMILY_ORDER: string[] = [
  "stack",
  "lightgbm",
  "xgboost",
  "rf",
  "lstm",
];

export function AlgorithmComparisonTable(_props: Props = {}) {
  const { data, isLoading } = useLeaderboard();
  const [crop, setCrop] = useState<string>("");

  const allRows = data?.rows ?? [];
  const crops = useMemo(
    () => [...new Set(allRows.map((r) => r.crop))].sort(),
    [allRows],
  );
  const effectiveCrop = crop || crops[0] || "";

  const byFamily = useMemo(() => {
    const out: Record<string, LeaderboardRow> = {};
    for (const r of allRows) {
      if (r.crop !== effectiveCrop) continue;
      out[r.family] = r;
    }
    return out;
  }, [allRows, effectiveCrop]);

  if (isLoading) {
    return (
      <Card className="flex h-32 items-center justify-center">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </Card>
    );
  }
  if (allRows.length === 0) {
    return (
      <Card className="p-6 text-center text-sm text-muted-foreground">
        Лідерборд порожній. Запустіть{" "}
        <code className="rounded bg-muted px-1">scripts/train_yield_models_v3.py</code>{" "}
        +{" "}
        <code className="rounded bg-muted px-1">scripts/evaluate_models.py</code>.
      </Card>
    );
  }

  // Compute shared kfold-R² extents across this crop's families so the
  // sparkline scaling is consistent within the table.
  const kfoldValues: number[] = [];
  for (const f of FAMILY_ORDER) {
    const r = byFamily[f];
    if (r?.kfold_r2_mean != null) kfoldValues.push(r.kfold_r2_mean);
  }
  const kfoldMin = kfoldValues.length ? Math.min(...kfoldValues) : 0;
  const kfoldMax = kfoldValues.length ? Math.max(...kfoldValues) : 1;

  return (
    <Card className="overflow-x-auto p-0">
      <div className="flex items-center justify-between border-b bg-muted/30 px-4 py-2 text-xs">
        <label className="inline-flex items-center gap-1">
          <span className="text-muted-foreground">Культура:</span>
          <select
            className="rounded border bg-background px-1 py-0.5"
            value={effectiveCrop}
            onChange={(e) => setCrop(e.target.value)}
          >
            {crops.map((c) => (
              <option key={c} value={c}>
                {CROP_LABEL_UK[c] ?? c}
              </option>
            ))}
          </select>
        </label>
        <span className="text-muted-foreground">
          {Object.keys(byFamily).length} з 6 моделей натреновано
        </span>
      </div>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b text-xs text-muted-foreground">
            <th className="px-3 py-2 text-left font-medium">Модель</th>
            <th className="px-3 py-2 text-right font-medium">Test R²</th>
            <th className="px-3 py-2 text-right font-medium">MAE</th>
            <th className="px-3 py-2 text-right font-medium">MAPE %</th>
            <th className="px-3 py-2 text-right font-medium">LOOCV R²</th>
            <th className="px-3 py-2 text-left font-medium" title="RepeatedKFold(5×3) mean ± σ">
              CV-variance
            </th>
            <th className="px-3 py-2 text-right font-medium">n train</th>
          </tr>
        </thead>
        <tbody>
          {FAMILY_ORDER.map((family) => {
            const r = byFamily[family];
            const isRecommended = family === "stack" && r != null;
            return (
              <tr
                key={family}
                className={cn(
                  "border-b last:border-b-0",
                  isRecommended && "bg-purple-50/50 dark:bg-purple-950/10",
                )}
              >
                <td className="px-3 py-2 font-medium">
                  {FAMILY_LABEL[family] ?? family}
                  {isRecommended && (
                    <span className="ml-1 rounded bg-purple-100 px-1 py-0 text-[10px] font-medium text-purple-700 dark:bg-purple-900 dark:text-purple-200">
                      default
                    </span>
                  )}
                </td>
                <NumCell v={r?.test_r2} digits={3} emphasis={r?.test_r2 != null && r.test_r2 > 0.5} />
                <NumCell v={r?.test_mae} digits={3} />
                <NumCell v={r?.test_mape} digits={1} suffix="%" />
                <NumCell v={r?.loocv_r2} digits={3} />
                <td className="px-3 py-2">
                  {r?.kfold_r2_mean != null ? (
                    <CVSparkline
                      mean={r.kfold_r2_mean}
                      std={r.kfold_r2_std ?? 0}
                      vmin={kfoldMin}
                      vmax={kfoldMax}
                    />
                  ) : (
                    <span className="text-xs text-muted-foreground">—</span>
                  )}
                </td>
                <td className="px-3 py-2 text-right text-xs text-muted-foreground">
                  {r?.n_train ?? "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </Card>
  );
}

function NumCell({
  v,
  digits,
  suffix,
  emphasis,
}: {
  v: number | null | undefined;
  digits: number;
  suffix?: string;
  emphasis?: boolean;
}) {
  if (v == null) {
    return <td className="px-3 py-2 text-right text-muted-foreground">—</td>;
  }
  return (
    <td
      className={cn(
        "px-3 py-2 text-right tabular-nums",
        emphasis && "font-semibold text-emerald-600",
      )}
    >
      {v.toFixed(digits)}
      {suffix}
    </td>
  );
}

/**
 * Inline SVG sparkline showing mean ± σ from RepeatedKFold(5×3).
 *
 * 60-pixel wide horizontal bar normalised to the crop's local kfold
 * range, with a darker filled rectangle for [mean − σ, mean + σ].
 * Compact enough to live inside a table cell yet visually decoding
 * "wide error bar = unstable model" at a glance.
 */
function CVSparkline({
  mean,
  std,
  vmin,
  vmax,
}: {
  mean: number;
  std: number;
  vmin: number;
  vmax: number;
}) {
  const W = 80;
  const H = 14;
  const range = Math.max(vmax - vmin, 0.01);

  const meanX = ((mean - vmin) / range) * W;
  const loX = ((mean - std - vmin) / range) * W;
  const hiX = ((mean + std - vmin) / range) * W;

  return (
    <span className="inline-flex items-center gap-1">
      <svg width={W} height={H} className="overflow-visible">
        {/* Range background bar */}
        <rect x={0} y={H / 2 - 1} width={W} height={2} fill="#e5e7eb" rx={1} />
        {/* ±σ band */}
        <rect
          x={Math.max(0, Math.min(W, loX))}
          y={H / 2 - 4}
          width={Math.max(0, Math.min(W, hiX) - Math.max(0, loX))}
          height={8}
          fill="#3b82f6"
          fillOpacity={0.35}
          rx={1}
        />
        {/* Mean dot */}
        <circle cx={meanX} cy={H / 2} r={3} fill="#3b82f6" />
      </svg>
      <span className="font-mono text-[10px] text-muted-foreground">
        {mean.toFixed(3)}
        {std > 0 && <span> ±{std.toFixed(3)}</span>}
      </span>
    </span>
  );
}
