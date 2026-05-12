import { useMemo, useState } from "react";
import { Loader2 } from "lucide-react";
import {
  CartesianGrid,
  Label,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

import { useEvaluationV3 } from "@/hooks/useMethodology";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

/**
 * Per-crop predicted-vs-actual scatter with 1:1 reference line.
 *
 * Each dot is one (oblast, year) sample from the 2023 test split. The
 * 1:1 diagonal is the "perfect prediction" line — points above it
 * mean over-prediction, below = under-prediction. Hovering shows the
 * oblast + the absolute residual.
 *
 * Reads `body.pred_vs_actual` from /api/methodology/evaluation_v3 —
 * the JSON caps each entry at 500 points so rendering stays smooth.
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

export function PerCropResidualScatter() {
  const { data, isLoading } = useEvaluationV3();
  const allCrops = useMemo(
    () => (data ? Object.keys(data.crops).sort() : []),
    [data],
  );
  const [crop, setCrop] = useState<string>("");
  const [family, setFamily] = useState<string>("stack");

  const effectiveCrop = crop || allCrops[0] || "";
  const cropBody = effectiveCrop ? data?.crops[effectiveCrop] : undefined;
  const allFamilies = cropBody ? Object.keys(cropBody) : [];
  const effectiveFamily = allFamilies.includes(family)
    ? family
    : (allFamilies[0] ?? "");
  const body = cropBody?.[effectiveFamily];

  const points = useMemo(() => {
    const raw = body?.pred_vs_actual ?? [];
    return raw.map((p) => ({
      actual: p.actual,
      predicted: p.predicted,
      oblast: p.oblast,
      iso_3166_2: p.iso_3166_2,
      residual: p.predicted - p.actual,
    }));
  }, [body]);

  const { vmin, vmax } = useMemo(() => {
    if (points.length === 0) return { vmin: 0, vmax: 10 };
    const all = points.flatMap((p) => [p.actual, p.predicted]);
    const lo = Math.min(...all);
    const hi = Math.max(...all);
    // 5 % padding so the 1:1 line doesn't touch the corners.
    const pad = (hi - lo) * 0.05;
    return { vmin: lo - pad, vmax: hi + pad };
  }, [points]);

  if (isLoading) {
    return (
      <Card>
        <CardContent className="flex h-64 items-center justify-center">
          <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    );
  }
  if (!data || allCrops.length === 0) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-sm text-muted-foreground">
          Дані pred-vs-actual ще не згенеровані. Запустіть{" "}
          <code className="rounded bg-muted px-1">scripts/evaluate_models.py</code>.
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="space-y-2">
        <CardTitle className="text-base">
          Прогнозоване vs фактичне (2023 test-split)
        </CardTitle>
        <div className="flex flex-wrap items-center gap-3 text-xs">
          <label className="inline-flex items-center gap-1">
            <span className="text-muted-foreground">Культура:</span>
            <select
              className="rounded border bg-background px-1 py-0.5"
              value={effectiveCrop}
              onChange={(e) => setCrop(e.target.value)}
            >
              {allCrops.map((c) => (
                <option key={c} value={c}>{CROP_LABEL_UK[c] ?? c}</option>
              ))}
            </select>
          </label>
          <label className="inline-flex items-center gap-1">
            <span className="text-muted-foreground">Модель:</span>
            <select
              className="rounded border bg-background px-1 py-0.5"
              value={effectiveFamily}
              onChange={(e) => setFamily(e.target.value)}
            >
              {allFamilies.map((f) => (
                <option key={f} value={f}>{f}</option>
              ))}
            </select>
          </label>
          {body?.test?.r2 != null && (
            <span className="text-muted-foreground">
              Test R²:{" "}
              <span className="font-medium text-foreground">
                {body.test.r2.toFixed(3)}
              </span>
            </span>
          )}
          {body?.test?.mape != null && (
            <span className="text-muted-foreground">
              MAPE:{" "}
              <span className="font-medium text-foreground">
                {body.test.mape.toFixed(1)}%
              </span>
            </span>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {points.length === 0 ? (
          <div className="py-8 text-center text-sm text-muted-foreground">
            Для цієї пари (модель × культура) немає даних.
          </div>
        ) : (
          <>
            <ResponsiveContainer width="100%" height={380}>
              <ScatterChart margin={{ top: 8, right: 20, bottom: 12, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  type="number"
                  dataKey="actual"
                  name="actual"
                  domain={[vmin, vmax]}
                  tick={{ fontSize: 11 }}
                >
                  <Label
                    value="фактичне (т/га)"
                    position="insideBottom"
                    offset={-2}
                    style={{ fontSize: 11 }}
                  />
                </XAxis>
                <YAxis
                  type="number"
                  dataKey="predicted"
                  name="predicted"
                  domain={[vmin, vmax]}
                  tick={{ fontSize: 11 }}
                >
                  <Label
                    value="прогноз (т/га)"
                    angle={-90}
                    position="insideLeft"
                    style={{ fontSize: 11 }}
                  />
                </YAxis>
                <ZAxis type="category" dataKey="oblast" range={[60, 60]} />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  content={({ active, payload }) => {
                    if (!active || !payload?.length) return null;
                    const p = payload[0].payload as (typeof points)[number];
                    return (
                      <div className="rounded border bg-popover p-2 text-xs shadow-md">
                        <div className="font-medium">{p.oblast}</div>
                        <div className="text-muted-foreground">{p.iso_3166_2}</div>
                        <div className="mt-1">
                          actual: <span className="font-mono">{p.actual.toFixed(3)}</span>
                        </div>
                        <div>
                          predicted: <span className="font-mono">{p.predicted.toFixed(3)}</span>
                        </div>
                        <div
                          className={
                            p.residual >= 0
                              ? "text-emerald-600"
                              : "text-red-600"
                          }
                        >
                          residual: <span className="font-mono">{p.residual >= 0 ? "+" : ""}{p.residual.toFixed(3)}</span>
                        </div>
                      </div>
                    );
                  }}
                />
                <ReferenceLine
                  segment={[
                    { x: vmin, y: vmin },
                    { x: vmax, y: vmax },
                  ]}
                  stroke="#94a3b8"
                  strokeDasharray="5 5"
                  ifOverflow="extendDomain"
                />
                <Scatter data={points} fill="#5e7d36" fillOpacity={0.7} />
              </ScatterChart>
            </ResponsiveContainer>
            <p className="mt-2 text-xs text-muted-foreground">
              Діагональ — лінія ідеального прогнозу. Точки над лінією — модель
              переоцінює врожай; під лінією — недооцінює. Систематичний зсув
              від діагоналі = bias моделі.
            </p>
          </>
        )}
      </CardContent>
    </Card>
  );
}
