import { useMemo, useState } from "react";
import { Loader2 } from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { useEvaluationV3 } from "@/hooks/useMethodology";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

/**
 * Learning curve — test R² as a function of training-sample size.
 *
 * Reads `learning_curve` from each (crop, family) eval entry; plots
 * each model family as one line. Lets the operator answer the
 * bias-variance question: "would more data still help?" If a curve
 * has plateaued, more data won't move the needle; if it's still
 * climbing at 100 %, we're data-limited.
 *
 * Source: evaluate_models.py emits points at fractions 0.25, 0.50,
 * 0.75, 1.0 of the train pool (≤2022); test is the fixed 2023 split.
 */
const FAMILY_COLOUR: Record<string, string> = {
  rf:        "#5e7d36",
  xgboost:   "#3b82f6",
  lightgbm:  "#a16207",
  catboost:  "#dc2626",
  stack:     "#9333ea",
  lstm:      "#0ea5e9",
};

export function LearningCurves() {
  const { data, isLoading } = useEvaluationV3();
  const allCrops = useMemo(
    () => (data ? Object.keys(data.crops).sort() : []),
    [data],
  );
  const [crop, setCrop] = useState<string>("");

  const effectiveCrop = crop || allCrops[0] || "";
  const cropBody = effectiveCrop ? data?.crops[effectiveCrop] : undefined;

  const chartData = useMemo(() => {
    if (!cropBody) return [];
    const buckets = new Map<number, Record<string, number | null>>();
    for (const [family, body] of Object.entries(cropBody)) {
      const curve = body?.learning_curve ?? [];
      for (const p of curve) {
        const entry = buckets.get(p.n_train) ?? { n_train: p.n_train };
        entry[family] = p.test_r2;
        buckets.set(p.n_train, entry);
      }
    }
    return [...buckets.values()].sort((a, b) =>
      Number(a.n_train) - Number(b.n_train),
    );
  }, [cropBody]);

  const familiesInChart = useMemo(() => {
    if (!cropBody) return [];
    return Object.entries(cropBody)
      .filter(([, body]) => (body?.learning_curve ?? []).length > 0)
      .map(([family]) => family);
  }, [cropBody]);

  if (isLoading) {
    return (
      <Card>
        <CardContent className="flex h-64 items-center justify-center">
          <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    );
  }
  if (allCrops.length === 0) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-sm text-muted-foreground">
          Learning curves ще не обчислені. Запустіть{" "}
          <code className="rounded bg-muted px-1">scripts/evaluate_models.py</code>{" "}
          (без флага <code>--skip-learning-curve</code>).
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="space-y-2">
        <CardTitle className="text-base">Криві навчання (Learning curves)</CardTitle>
        <div className="flex items-center gap-2 text-xs">
          <label className="inline-flex items-center gap-1">
            <span className="text-muted-foreground">Культура:</span>
            <select
              className="rounded border bg-background px-1 py-0.5"
              value={effectiveCrop}
              onChange={(e) => setCrop(e.target.value)}
            >
              {allCrops.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </label>
        </div>
      </CardHeader>
      <CardContent>
        {chartData.length === 0 || familiesInChart.length === 0 ? (
          <div className="py-8 text-center text-sm text-muted-foreground">
            Дані learning curves порожні для цієї культури.
          </div>
        ) : (
          <>
            <ResponsiveContainer width="100%" height={360}>
              <LineChart
                data={chartData}
                margin={{ top: 12, right: 20, bottom: 12, left: 4 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="n_train"
                  type="number"
                  tick={{ fontSize: 11 }}
                  label={{ value: "n train", position: "insideBottom", offset: -2, style: { fontSize: 11 } }}
                />
                <YAxis
                  tick={{ fontSize: 11 }}
                  label={{ value: "test R²", angle: -90, position: "insideLeft", style: { fontSize: 11 } }}
                />
                <Tooltip
                  formatter={(v) =>
                    typeof v === "number" ? v.toFixed(3) : "—"
                  }
                  labelFormatter={(n) => `n_train = ${n}`}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {familiesInChart.map((family) => (
                  <Line
                    key={family}
                    type="monotone"
                    dataKey={family}
                    name={family}
                    stroke={FAMILY_COLOUR[family] ?? "#666"}
                    strokeWidth={2}
                    dot={{ r: 3 }}
                    connectNulls
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
            <p className="mt-2 text-xs text-muted-foreground">
              Test R² при підвиборках 25/50/75/100 % з train+val (≤2022); 2023 — фіксований test-split.
              Якщо крива ще піднімається на 100 % — модель data-limited, додавання даних допоможе.
            </p>
          </>
        )}
      </CardContent>
    </Card>
  );
}
