import { useMemo } from "react";
import { Loader2 } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { useLeaderboard } from "@/hooks/useMethodology";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

/**
 * Quantile-head calibration check for the XGBoost q=0.05/q=0.95 outputs.
 *
 * Plots two metrics per (crop) showing how well-calibrated and sharp
 * the 90 % prediction intervals are:
 *
 *  1. **Interval coverage @ 90 %** — fraction of test samples whose
 *     true value lands inside the [q05, q95] interval. Should be ≈0.9
 *     if calibrated; reference line at 0.9.
 *  2. **Pinball loss (sum q05 + q95)** — sharpness; lower = tighter
 *     intervals without losing coverage. Koenker & Bassett (1978).
 *
 * Only XGBoost has quantile heads in v3 trainer; we filter the
 * leaderboard rows to that family.
 */
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

export function CalibrationPlot() {
  const { data, isLoading } = useLeaderboard();

  const chartData = useMemo(() => {
    const rows = (data?.rows ?? []).filter(
      (r) =>
        r.family === "xgboost" &&
        r.interval_coverage_90pct != null &&
        (r.pinball_q05 != null || r.pinball_q95 != null),
    );
    return rows
      .map((r) => ({
        crop: CROP_LABEL_UK[r.crop] ?? r.crop,
        coverage: r.interval_coverage_90pct ?? 0,
        // Sum of pinball losses (q05 + q95) — single sharpness number.
        pinball_sum:
          (r.pinball_q05 ?? 0) + (r.pinball_q95 ?? 0),
        // Visual flag — within ±5 % of target coverage is "calibrated".
        coverageWellCalibrated:
          r.interval_coverage_90pct != null &&
          Math.abs(r.interval_coverage_90pct - 0.9) < 0.05,
      }))
      .sort((a, b) => b.coverage - a.coverage);
  }, [data]);

  if (isLoading) {
    return (
      <Card>
        <CardContent className="flex h-64 items-center justify-center">
          <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    );
  }
  if (!data || chartData.length === 0) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-sm text-muted-foreground">
          Calibration не доступний — потрібен XGBoost з quantile heads
          (q=0.05 / q=0.95) у{" "}
          <code className="rounded bg-muted px-1">model_metrics_v3.json</code>.
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">
          Калібрування quantile heads
          <span className="ml-2 text-xs font-normal text-muted-foreground">
            XGBoost q=0.05 / q=0.95 — Koenker &amp; Bassett 1978
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart
            data={chartData}
            margin={{ top: 8, right: 30, bottom: 12, left: 20 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="crop"
              tick={{ fontSize: 10 }}
              angle={-30}
              textAnchor="end"
              height={70}
              interval={0}
            />
            <YAxis
              yAxisId="left"
              orientation="left"
              tick={{ fontSize: 10 }}
              label={{
                value: "interval coverage @90%",
                angle: -90,
                position: "insideLeft",
                offset: 10,
                style: { fontSize: 10 },
              }}
              domain={[0, 1]}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              tick={{ fontSize: 10 }}
              label={{
                value: "pinball loss (q05+q95)",
                angle: 90,
                position: "insideRight",
                offset: 10,
                style: { fontSize: 10 },
              }}
            />
            <Tooltip
              formatter={(v, key) =>
                typeof v === "number"
                  ? key === "coverage"
                    ? `${(v * 100).toFixed(1)}%`
                    : v.toFixed(3)
                  : "—"
              }
              contentStyle={{ fontSize: 11 }}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <ReferenceLine
              yAxisId="left"
              y={0.9}
              stroke="#dc2626"
              strokeDasharray="5 5"
              label={{
                value: "ціль 0.9",
                position: "insideTopRight",
                style: { fontSize: 10, fill: "#dc2626" },
              }}
            />
            <Bar
              yAxisId="left"
              dataKey="coverage"
              name="coverage @90%"
              fill="#3b82f6"
            />
            <Bar
              yAxisId="right"
              dataKey="pinball_sum"
              name="pinball sum"
              fill="#a16207"
            />
          </BarChart>
        </ResponsiveContainer>
        <p className="mt-2 text-xs text-muted-foreground">
          <strong>Coverage</strong> (синій) — частка true values які лежать
          у 90 %-інтервалі (ціль ≈ 0.9, червона лінія).{" "}
          <strong>Pinball loss</strong> (коричневий) — sharpness інтервалу
          (нижче — тісніше). Ідеал: coverage ≈ 0.9 + низький pinball =
          well-calibrated &amp; sharp prediction intervals (Gneiting,
          Balabdaoui &amp; Raftery 2007).
        </p>
      </CardContent>
    </Card>
  );
}
