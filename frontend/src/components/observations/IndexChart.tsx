import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useTranslation } from "react-i18next";
import type { IndexName, ObservationRead } from "@/api/observations";

interface IndexChartProps {
  observations: ObservationRead[];
  index: IndexName;
  selectedDate: string | null;
  onPointClick: (date: string) => void;
}

interface ChartPoint {
  date: string;
  value: number | null;
  cloud: number | null;
}

function valueOf(obs: ObservationRead, index: IndexName): number | null {
  switch (index) {
    case "ndvi":
      return obs.ndvi_mean;
    case "evi":
      return obs.evi_mean;
    case "ndwi":
      return obs.ndwi_mean;
    case "savi":
      return obs.savi_mean;
  }
}

// Tuned Y-axis ranges per index.
const Y_DOMAIN: Record<IndexName, [number, number]> = {
  ndvi: [-0.2, 1],
  evi: [-0.5, 1],
  ndwi: [-1, 1],
  savi: [-0.5, 1.2],
};

const INDEX_COLOR: Record<IndexName, string> = {
  ndvi: "#5e7d36",
  evi: "#7d9c49",
  ndwi: "#3b82f6",
  savi: "#a16207",
};

export function IndexChart({
  observations,
  index,
  selectedDate,
  onPointClick,
}: IndexChartProps) {
  const { t } = useTranslation();

  const data: ChartPoint[] = useMemo(
    () =>
      observations.map((o) => ({
        date: o.observed_on,
        value: valueOf(o, index),
        cloud: o.cloud_cover,
      })),
    [observations, index],
  );

  if (data.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
        {t("observations.empty.title")}
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart
        data={data}
        margin={{ top: 10, right: 20, left: 0, bottom: 10 }}
        onClick={(state) => {
          // Recharts gives an activeLabel (the date) when clicked on a data point.
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const label = (state as any)?.activeLabel as string | undefined;
          if (label) onPointClick(label);
        }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis
          dataKey="date"
          tick={{ fontSize: 11 }}
          tickFormatter={(d) => d.slice(2, 10)}
          minTickGap={30}
        />
        <YAxis
          domain={Y_DOMAIN[index]}
          tick={{ fontSize: 11 }}
          width={45}
        />
        <Tooltip
          content={({ active, payload, label }) => {
            if (!active || !payload?.[0]) return null;
            const p = payload[0].payload as ChartPoint;
            return (
              <div className="rounded-md border bg-popover px-3 py-2 text-xs shadow-md">
                <div className="font-semibold">{label}</div>
                <div className="mt-1 text-muted-foreground">
                  {p.value !== null && p.value !== undefined
                    ? `${t(`indices.${index}`)}: ${p.value.toFixed(3)}`
                    : t("observations.cloudGap")}
                </div>
              </div>
            );
          }}
        />
        <Line
          type="monotone"
          dataKey="value"
          stroke={INDEX_COLOR[index]}
          strokeWidth={2}
          connectNulls={false}
          dot={{ r: 3, cursor: "pointer" }}
          activeDot={{ r: 6 }}
          isAnimationActive={false}
        />
        {selectedDate && (
          <ReferenceDot
            x={selectedDate}
            y={data.find((d) => d.date === selectedDate)?.value ?? 0}
            r={7}
            fill="#dc2626"
            stroke="#fff"
            strokeWidth={2}
          />
        )}
      </LineChart>
    </ResponsiveContainer>
  );
}
