import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowRight, CloudRain, Sun, Thermometer } from "lucide-react";

import type { FieldWeather } from "@/api/dashboard";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface Props {
  weatherByField: FieldWeather[];
}

/**
 * Compact portfolio-wide weather glance on the dashboard. Three aggregated
 * numbers — hottest temp, biggest precip sum, max heat-stress days across
 * fields — plus a Details button that opens the dedicated /weather page
 * where the user can drill into any single field's 14-day forecast,
 * day-strip, and advisories.
 *
 * Replaces the old per-field dropdown card; the picker now lives on the
 * full /weather page so we don't duplicate the experience.
 */
export function WeatherSummaryCard({ weatherByField }: Props) {
  const navigate = useNavigate();
  const summary = useMemo(() => aggregate(weatherByField), [weatherByField]);

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-base">Погода 7 днів</CardTitle>
        <Button
          variant="ghost"
          size="sm"
          className="h-7 text-xs"
          onClick={() => navigate("/weather")}
        >
          Деталі
          <ArrowRight className="h-3 w-3" />
        </Button>
      </CardHeader>
      <CardContent>
        {summary ? (
          <>
            <div className="grid grid-cols-3 gap-2">
              <Stat
                icon={<Thermometer className="h-4 w-4 text-red-500" />}
                label="Макс."
                value={`${summary.tempMax.toFixed(0)}°C`}
              />
              <Stat
                icon={<CloudRain className="h-4 w-4 text-blue-500" />}
                label="Опади"
                value={`${summary.precipMax.toFixed(1)} мм`}
              />
              <Stat
                icon={<Sun className="h-4 w-4 text-amber-500" />}
                label="Спека"
                value={`${summary.heatDays} дн.`}
                emphasize={summary.heatDays >= 3}
              />
            </div>
            <p className="mt-3 text-[11px] text-muted-foreground">
              {summary.fieldCount === 1
                ? summary.firstFieldName
                : `Узагальнено для ${summary.fieldCount} полів. Натисніть «Деталі» — деталізація по кожному.`}
            </p>
          </>
        ) : (
          <div className="text-sm text-muted-foreground">
            Прогноз поки не зібрано. Додайте поля — фоновий job завантажить
            дані з Open-Meteo.
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function Stat({
  icon,
  label,
  value,
  emphasize = false,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  emphasize?: boolean;
}) {
  return (
    <div
      className={cn(
        "rounded-md border p-2",
        emphasize ? "border-amber-300 bg-amber-50" : "bg-card",
      )}
    >
      <div className="flex items-center gap-1 text-[10px] text-muted-foreground">
        {icon}
        {label}
      </div>
      <div className="mt-0.5 text-lg font-semibold tabular-nums">{value}</div>
    </div>
  );
}

function aggregate(fields: FieldWeather[]):
  | {
      tempMax: number;
      precipMax: number;
      heatDays: number;
      fieldCount: number;
      firstFieldName: string;
    }
  | null {
  const withDays = fields.filter((f) => f.days.length > 0);
  if (withDays.length === 0) return null;

  let tempMax = -Infinity;
  let precipMax = 0;
  let heatDaysMax = 0;
  for (const f of withDays) {
    if (f.temp_max_7d !== null) tempMax = Math.max(tempMax, f.temp_max_7d);
    if (f.precip_sum_7d !== null) precipMax = Math.max(precipMax, f.precip_sum_7d);
    heatDaysMax = Math.max(heatDaysMax, f.heat_stress_days_7d);
  }
  return {
    tempMax: tempMax === -Infinity ? 0 : tempMax,
    precipMax,
    heatDays: heatDaysMax,
    fieldCount: withDays.length,
    firstFieldName: withDays[0].field_name,
  };
}
