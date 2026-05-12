import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import {
  ArrowRight,
  Cloud,
  CloudDrizzle,
  CloudRain,
  Sun,
  Thermometer,
} from "lucide-react";

import type { FieldWeather, WeatherDay } from "@/api/dashboard";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CROP_COLORS } from "@/lib/colors";
import { cn } from "@/lib/utils";

interface Props {
  weatherByField: FieldWeather[];
}

/**
 * Compact portfolio-wide weather glance on the dashboard. Three aggregated
 * tiles (hottest temp / biggest precip sum / max heat-stress days) sit on
 * top, followed by a per-field list (icon + crop chip + name + avg temp +
 * min/max range). The list inherits the dashboard's `FilterBar` crop
 * filter: the backend filters `fields` before computing `weather_by_field`,
 * so toggling crops at the top of /dashboard narrows the per-field list
 * here automatically — no separate crop picker.
 *
 * The "Деталі →" button navigates to the dedicated /weather page for a
 * full 14-day per-field breakdown.
 */

// How many per-field rows fit comfortably under the 3 aggregate tiles
// without forcing the card taller than its dashboard grid neighbours.
// Larger portfolios get a "+ ще N полів" hint linking to /weather.
const MAX_FIELD_ROWS = 6;


export function WeatherSummaryCard({ weatherByField }: Props) {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const summary = useMemo(() => aggregate(weatherByField), [weatherByField]);

  // Hottest-first ordering — same heuristic that drives the
  // heat-stress tile, so the per-field list opens with the fields
  // most worth glancing at.
  const sortedFields = useMemo(
    () =>
      [...weatherByField]
        .filter((f) => f.days.length > 0)
        .sort(
          (a, b) =>
            (b.temp_max_7d ?? -Infinity) - (a.temp_max_7d ?? -Infinity),
        ),
    [weatherByField],
  );

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

            {sortedFields.length > 0 && (
              <ul className="mt-3 space-y-1.5 border-t pt-3">
                {sortedFields.slice(0, MAX_FIELD_ROWS).map((f) => (
                  <li
                    key={f.field_id}
                    className="flex items-center gap-2 text-xs"
                  >
                    <WeatherIcon days={f.days} />
                    <span
                      className="h-2 w-2 rounded-sm shrink-0"
                      style={{ background: CROP_COLORS[f.crop_type] }}
                      aria-hidden
                    />
                    <span className="min-w-0 flex-1 truncate font-medium">
                      {f.field_name}
                      <span className="ml-1 font-normal text-muted-foreground">
                        · {t(`fields.crops.${f.crop_type}`)}
                      </span>
                    </span>
                    <span className="tabular-nums text-muted-foreground shrink-0">
                      Серед.{" "}
                      {f.temp_avg_7d != null
                        ? `${f.temp_avg_7d.toFixed(0)}°C`
                        : "—"}
                    </span>
                    <span className="tabular-nums shrink-0">
                      ↑
                      {f.temp_max_7d != null
                        ? `${f.temp_max_7d.toFixed(0)}°`
                        : "—"}
                      /↓
                      {f.temp_min_7d != null
                        ? `${f.temp_min_7d.toFixed(0)}°`
                        : "—"}
                    </span>
                  </li>
                ))}
              </ul>
            )}

            <p className="mt-3 text-[11px] text-muted-foreground">
              {sortedFields.length > MAX_FIELD_ROWS
                ? `Показано ${MAX_FIELD_ROWS} з ${sortedFields.length} полів — натисніть «Деталі», щоб побачити всі.`
                : `Узагальнено для ${summary.fieldCount} ${pluralFields(summary.fieldCount)}.`}
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


/**
 * Pick a single dominant condition for the 7-day forecast. This is a
 * portfolio-overview indicator, not an authoritative forecast — the 3
 * aggregate tiles above the list remain the primary numerical signal.
 *
 * Decision order (first hit wins):
 *   1. Any day with > 5 mm precip   → heavy rain
 *   2. Total precip > 5 mm          → drizzle / showers
 *   3. Every day max-temp > 28 °C   → sunny / hot
 *   4. Otherwise                    → generic cloudy
 *
 * Thresholds chosen to match the agronomic copy on /weather (the same
 * `rain > 10 mm/day = heavy` and `temp_max > 30 °C = heat-stress` bands
 * used by `weather_advice.py`, just slightly relaxed for an "overview at
 * a glance" icon rather than an alert trigger).
 */
function WeatherIcon({ days }: { days: WeatherDay[] }) {
  const heavyRain = days.some((d) => (d.precip_mm ?? 0) > 5);
  const totalPrecip = days.reduce((s, d) => s + (d.precip_mm ?? 0), 0);
  const allHot = days.every(
    (d) => d.temp_max_c != null && d.temp_max_c > 28,
  );
  if (heavyRain) return <CloudRain className="h-3.5 w-3.5 shrink-0 text-blue-500" />;
  if (totalPrecip > 5) return <CloudDrizzle className="h-3.5 w-3.5 shrink-0 text-sky-500" />;
  if (allHot) return <Sun className="h-3.5 w-3.5 shrink-0 text-amber-500" />;
  return <Cloud className="h-3.5 w-3.5 shrink-0 text-slate-400" />;
}


function pluralFields(n: number): string {
  // Ukrainian plural for "поле": 1 → поля, 2-4 → полів, ≥5 → полів.
  // Genitive plural is "полів" for everything except 1 (which uses the
  // genitive singular "поля" after a numeral — agronomic context).
  // Simple binary split is good enough for the footer one-liner.
  return n === 1 ? "поля" : "полів";
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
