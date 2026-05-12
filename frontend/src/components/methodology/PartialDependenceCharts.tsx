import { useMemo, useState } from "react";
import { Loader2 } from "lucide-react";
import {
  CartesianGrid,
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
 * 1-D Partial Dependence Plots (PDP) for the top-3 features per
 * (crop, family). Each curve shows the marginal effect of one feature
 * on the model's prediction, holding all others fixed at their
 * empirical distribution — Friedman (2001).
 *
 * Three side-by-side LineCharts. PDP curves are pre-computed in
 * `evaluate_models.py` so the frontend just renders.
 *
 * Notes:
 *   - Stacked ensemble has no PDP (sklearn `partial_dependence`
 *     reports meta-input dimensions, not original 17 features).
 *     UI shows a friendly "not available" message.
 *   - For features whose value range is degenerate (constant), the
 *     curve is a single horizontal line — still informative.
 */
const FEATURE_LABEL: Record<string, string> = {
  ndvi_peak: "NDVI peak",
  ndvi_peak_week: "NDVI peak week",
  ndvi_mean_may: "NDVI May",
  ndvi_mean_june: "NDVI June",
  ndvi_mean_july: "NDVI July",
  ndvi_mean_august: "NDVI August",
  ndvi_integral: "NDVI integral",
  ndvi_std: "NDVI std",
  evi_peak: "EVI peak",
  ndwi_min: "NDWI min",
  savi_peak: "SAVI peak",
  precip_sum_apr_jul: "Precip apr-jul",
  temp_mean_apr_jul: "Temp apr-jul",
  heat_stress_days: "Heat-stress days",
  drought_dryspells: "Drought dryspells",
  centroid_lat: "Latitude",
  centroid_lon: "Longitude",
};

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

export function PartialDependenceCharts() {
  const { data, isLoading } = useEvaluationV3();
  const allCrops = useMemo(
    () => (data ? Object.keys(data.crops).sort() : []),
    [data],
  );
  const [crop, setCrop] = useState<string>("");
  // RF is the most PDP-stable default (deterministic, low variance curves).
  const [family, setFamily] = useState<string>("rf");

  const effectiveCrop = crop || allCrops[0] || "";
  const cropBody = effectiveCrop ? data?.crops[effectiveCrop] : undefined;
  const allFamilies = cropBody ? Object.keys(cropBody) : [];
  const effectiveFamily = allFamilies.includes(family)
    ? family
    : (allFamilies[0] ?? "");
  const body = cropBody?.[effectiveFamily];

  const pdp = body?.partial_dependence ?? {};
  const features = Object.keys(pdp);

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
          PDP не доступні — запустіть{" "}
          <code className="rounded bg-muted px-1">scripts/evaluate_models.py</code>.
        </CardContent>
      </Card>
    );
  }

  const isStack = effectiveFamily === "stack";
  const yieldUnit = effectiveCrop === "sugar_beet" || effectiveCrop === "potato" || effectiveCrop === "corn_silage"
    ? "t/ha (×10)"  // root + silage crops have much higher absolute values
    : "т/га";

  return (
    <Card>
      <CardHeader className="space-y-2">
        <CardTitle className="text-base">
          Partial Dependence Plots (Friedman 2001)
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
        </div>
      </CardHeader>
      <CardContent>
        {isStack ? (
          <div className="py-8 text-center text-sm text-muted-foreground">
            <strong>PDP не визначено для Stacked Ensemble.</strong>
            <br />
            Stack виходить через Ridge meta-learner поверх OOF-prediction базових моделей —
            sklearn повертає кривy для meta-input space (4 base-prediction виміри), не для
            оригінальних 17 фіч. Оберіть RF / XGBoost / LightGBM / CatBoost.
          </div>
        ) : features.length === 0 ? (
          <div className="py-8 text-center text-sm text-muted-foreground">
            PDP не обчислені для цієї пари. Перевірте лог{" "}
            <code className="rounded bg-muted px-1">evaluate_models.py</code>.
          </div>
        ) : (
          <>
            <div className="grid gap-4 sm:grid-cols-3">
              {features.map((feat) => {
                const body = pdp[feat];
                const chartData = body.grid.map((g: number, i: number) => ({
                  x: g,
                  pdp: body.pdp[i],
                }));
                return (
                  <div key={feat} className="rounded border p-2">
                    <p className="mb-1 text-xs font-medium">
                      {FEATURE_LABEL[feat] ?? feat}
                    </p>
                    <ResponsiveContainer width="100%" height={180}>
                      <LineChart
                        data={chartData}
                        margin={{ top: 4, right: 8, bottom: 18, left: 0 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                          dataKey="x"
                          type="number"
                          domain={["dataMin", "dataMax"]}
                          tickFormatter={(v) =>
                            typeof v === "number" ? v.toFixed(2) : String(v)
                          }
                          tick={{ fontSize: 9 }}
                        />
                        <YAxis
                          tickFormatter={(v) =>
                            typeof v === "number" ? v.toFixed(1) : String(v)
                          }
                          tick={{ fontSize: 9 }}
                          width={32}
                        />
                        <Tooltip
                          formatter={(v) =>
                            typeof v === "number" ? v.toFixed(3) : "—"
                          }
                          labelFormatter={(v) =>
                            typeof v === "number"
                              ? `x = ${v.toFixed(3)}`
                              : `x = ${v}`
                          }
                          contentStyle={{ fontSize: 11 }}
                        />
                        <Line
                          type="monotone"
                          dataKey="pdp"
                          stroke="#5e7d36"
                          strokeWidth={2}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                );
              })}
            </div>
            <p className="mt-3 text-xs text-muted-foreground">
              Кожен підгравік — як змінюється середній прогноз ({yieldUnit}) залежно від
              значення однієї фічі, при усередненні всіх інших. Top-3 фічі за
              permutation importance per (crop, family).
            </p>
          </>
        )}
      </CardContent>
    </Card>
  );
}
