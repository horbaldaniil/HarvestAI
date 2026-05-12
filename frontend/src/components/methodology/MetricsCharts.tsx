import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { ALL_CROPS } from "@/api/fields";
import type { ModelInfo } from "@/api/methodology";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

/**
 * Ukrainian display labels per crop slug. Mirrors `fields.crops.*` in
 * `lib/i18n.ts` — kept as a local map (instead of calling the i18n
 * hook) because `buildSeries` is a pure helper and is itself
 * memoised; we'd otherwise have to thread `t` through every call.
 * If you add a 14th crop, update both this map and the i18n file.
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

const FAMILY_COLORS: Record<string, string> = {
  xgboost: "#5e7d36",
  rf: "#3b82f6",
  lstm: "#a16207",
};

interface Props {
  models: ModelInfo[];
}

/**
 * Two bar charts side-by-side: test R² and test MAE, grouped by crop with
 * one bar per algorithm family. Higher R² is better; lower MAE is better.
 */
export function MetricsCharts({ models }: Props) {
  const r2Data = useMemo(() => buildSeries(models, "r2"), [models]);
  const maeData = useMemo(() => buildSeries(models, "mae"), [models]);

  if (models.length === 0 || r2Data.length === 0) {
    return (
      <Card>
        <CardContent className="py-10 text-center text-sm text-muted-foreground">
          Метрики ще не зібрано. Запустіть training scripts у `backend/scripts/`.
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <ChartCard title="R² на тест-сеті (вище — краще)" data={r2Data} valueFmt={(v) => v.toFixed(2)} />
      <ChartCard title="MAE, т/га (нижче — краще)" data={maeData} valueFmt={(v) => v.toFixed(2)} />
    </div>
  );
}

function ChartCard({
  title,
  data,
  valueFmt,
}: {
  title: string;
  data: Array<Record<string, unknown>>;
  valueFmt: (v: number) => string;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="crop" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} tickFormatter={valueFmt} />
            <Tooltip formatter={(v: number) => valueFmt(v)} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {(["xgboost", "rf", "lstm"] as const).map((fam) => (
              <Bar key={fam} dataKey={fam} fill={FAMILY_COLORS[fam]} name={fam.toUpperCase()}>
                {data.map((_, idx) => (
                  <Cell key={idx} fill={FAMILY_COLORS[fam]} />
                ))}
              </Bar>
            ))}
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

function buildSeries(models: ModelInfo[], metric: "r2" | "mae") {
  // Per (crop, family) pick the highest-version model and read .test[metric].
  const grid: Record<string, Record<string, number>> = {};
  for (const m of models) {
    const key = `${m.crop}_${m.family}`;
    const test = m.metrics?.test;
    if (!test) continue;
    const v = (test as unknown as Record<string, number>)[metric];
    if (v === undefined) continue;
    const prev = grid[key];
    if (!prev || (m.version > (prev._version as unknown as string))) {
      grid[key] = { value: v, _version: m.version as unknown as number };
    }
  }

  // Iterate the canonical 13-crop order from ALL_CROPS so the chart's
  // x-axis is stable across users / sessions / model-set sizes. Crops
  // without any model in `models` are filtered out to keep the chart
  // readable — with 13 crops × 3 families, every empty bar is noise.
  return ALL_CROPS.map((c) => {
    const row: Record<string, unknown> = { crop: CROP_LABEL_UK[c] ?? c };
    let hasAny = false;
    for (const fam of ["xgboost", "rf", "lstm"] as const) {
      const cell = grid[`${c}_${fam}`];
      if (cell !== undefined) {
        row[fam] = cell.value;
        hasAny = true;
      }
    }
    return hasAny ? row : null;
  }).filter((r): r is Record<string, unknown> => r !== null);
}
