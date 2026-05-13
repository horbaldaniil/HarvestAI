import { useMemo, useState } from "react";
import { Loader2 } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { useEvaluationV3 } from "@/hooks/useMethodology";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

/**
 * Global SHAP summary — mean |SHAP value| per feature for a chosen
 * (crop, family). The Lundberg & Lee (2017) bar plot, sorted descending.
 *
 * Lets a reader of the thesis quickly see WHICH inputs drive the
 * model's predictions on average. Concretely: for the synthetic v3
 * data the strongest driver is `centroid_lat` (the latitude divide
 * between Polissia and Steppe yields). With real Sentinel-2 features
 * that will shift toward NDVI peak / July mean — which is itself a
 * thesis-defensible result ("our pipeline picks the agronomically
 * relevant features when given good data").
 */
const HUMAN_LABEL: Record<string, string> = {
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
  precip_sum_apr_jul: "Precip apr–jul",
  temp_mean_apr_jul: "Temp apr–jul",
  heat_stress_days: "Heat-stress days",
  drought_dryspells: "Drought dryspells",
  centroid_lat: "Lat",
  centroid_lon: "Lon",
};

export function GlobalShapSummary() {
  const { data, isLoading, error } = useEvaluationV3();
  const allCrops = useMemo(
    () => (data ? Object.keys(data.crops).sort() : []),
    [data],
  );
  const [crop, setCrop] = useState<string>("");
  const [family, setFamily] = useState<string>("rf");
  const [mode, setMode] = useState<"shap" | "permutation">("shap");

  const effectiveCrop = crop || allCrops[0] || "";
  const cropBody = effectiveCrop ? data?.crops[effectiveCrop] : undefined;
  const allFamilies = cropBody ? Object.keys(cropBody) : [];
  const effectiveFamily = allFamilies.includes(family) ? family : (allFamilies[0] ?? "");
  const body = cropBody?.[effectiveFamily];

  const importance =
    mode === "shap"
      ? body?.global_shap ?? null
      : body?.permutation_importance ?? null;

  const chartData = useMemo(() => {
    if (!importance) return [];
    return Object.entries(importance)
      .map(([feat, value]) => ({
        feature: HUMAN_LABEL[feat] ?? feat,
        importance: Math.abs(value),
      }))
      .sort((a, b) => b.importance - a.importance);
  }, [importance]);

  if (isLoading) {
    return (
      <Card>
        <CardContent className="flex h-64 items-center justify-center">
          <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    );
  }
  if (error || !data) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-sm text-muted-foreground">
          Не вдалося завантажити SHAP-summary.
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="space-y-2">
        <CardTitle className="text-base">
          {mode === "shap" ? "Global SHAP — важливість фіч" : "Permutation importance"}
        </CardTitle>
        <div className="flex flex-wrap items-center gap-3 text-xs">
          <label className="inline-flex items-center gap-1">
            <span className="text-muted-foreground">Метод:</span>
            <select
              className="rounded border bg-background px-1 py-0.5"
              value={mode}
              onChange={(e) => setMode(e.target.value as "shap" | "permutation")}
            >
              <option value="shap">SHAP (Lundberg &amp; Lee 2017)</option>
              <option value="permutation">Permutation (Breiman 2001)</option>
            </select>
          </label>
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
        {chartData.length === 0 ? (
          <div className="py-8 text-center text-sm text-muted-foreground">
            Для цієї пари (модель × культура) {mode === "shap" ? "SHAP" : "permutation importance"}{" "}
            не обчислена. Stack-моделі (Ridge поверх OOF preds) не мають
            однозначної feature-importance — оберіть RF / XGBoost / LightGBM.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={420}>
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 4, right: 16, bottom: 4, left: 100 }}
            >
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis
                dataKey="feature"
                type="category"
                tick={{ fontSize: 11 }}
                width={100}
              />
              <Tooltip
                formatter={(v: number) => v.toFixed(4)}
                labelFormatter={(label) => `Фіча: ${label}`}
              />
              <Bar dataKey="importance" fill="#5e7d36" />
            </BarChart>
          </ResponsiveContainer>
        )}
        <p className="mt-2 text-xs text-muted-foreground">
          {mode === "shap"
            ? "Mean | SHAP value | — середній абсолютний вплив фічі на прогноз."
            : "Падіння R² при випадковій перестановці значень фічі — модель-агностична альтернатива."}
        </p>
      </CardContent>
    </Card>
  );
}
