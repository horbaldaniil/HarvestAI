import { format } from "date-fns";
import { uk } from "date-fns/locale";
import {
  Cloud,
  CloudRain,
  Sun,
  CloudSun,
  Cloudy,
  Droplets,
  Wind,
  Sprout,
} from "lucide-react";

import type { WeatherDay } from "@/api/weather";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

interface Props {
  days: WeatherDay[];
}

/**
 * Day-strip with three layers per day:
 *   1. Sky icon (☀️/⛅/☁️/🌧) driven by cloud_cover + precip
 *   2. Min/max vertical temperature bar with mean tick
 *   3. Precip bar (blue) at the bottom showing mm of rain
 *
 * Each day is wrapped in a Radix Tooltip showing the full detail (humidity,
 * wind, radiation, soil moisture).
 *
 * The bar uses a global min/max across the whole strip so the heights are
 * comparable day-to-day.
 */
export function WeatherDayStrip({ days }: Props) {
  if (days.length === 0) {
    return (
      <div className="rounded-md border bg-card p-4 text-center text-sm text-muted-foreground">
        Прогноз ще не зібрано.
      </div>
    );
  }

  // Compute global y-axis range so each day's bar shares the same scale.
  const allTemps = days
    .flatMap((d) => [d.temp_min_c, d.temp_max_c])
    .filter((v): v is number => v !== null);
  const yMin = Math.min(...allTemps, 0) - 2;
  const yMax = Math.max(...allTemps, 0) + 2;
  const yRange = yMax - yMin || 1;

  const allPrecip = days.map((d) => d.precip_mm ?? 0);
  const precipMax = Math.max(...allPrecip, 1);

  return (
    <div className="overflow-x-auto rounded-md border bg-card p-3">
      <div className="grid min-w-max gap-1" style={{ gridTemplateColumns: `repeat(${days.length}, minmax(64px, 1fr))` }}>
        {days.map((d) => (
          <DayColumn
            key={d.observed_on}
            day={d}
            yMin={yMin}
            yRange={yRange}
            precipMax={precipMax}
          />
        ))}
      </div>
    </div>
  );
}

function DayColumn({
  day,
  yMin,
  yRange,
  precipMax,
}: {
  day: WeatherDay;
  yMin: number;
  yRange: number;
  precipMax: number;
}) {
  const date = new Date(day.observed_on);
  const BAR_HEIGHT = 80;
  const tmin = day.temp_min_c;
  const tmax = day.temp_max_c;
  const tmean = day.temp_mean_c;

  // Convert temp → pixel offset from bottom.
  const yToPx = (v: number) => ((v - yMin) / yRange) * BAR_HEIGHT;
  const barBottom = tmin !== null ? yToPx(tmin) : 0;
  const barTop = tmax !== null ? yToPx(tmax) : BAR_HEIGHT;
  const meanY = tmean !== null ? yToPx(tmean) : null;

  const SkyIcon = pickSkyIcon(day);
  const precipPx = Math.round(((day.precip_mm ?? 0) / precipMax) * 28);

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div className="flex cursor-help flex-col items-center gap-1 rounded p-1 transition-colors hover:bg-accent/30">
          <div className="text-[10px] text-muted-foreground">
            {format(date, "EEE", { locale: uk })}
          </div>
          <div className="text-[10px] tabular-nums text-muted-foreground">
            {format(date, "d.MM", { locale: uk })}
          </div>
          <SkyIcon className={cn("h-4 w-4", skyColour(day))} />

          {/* Min/max temperature bar */}
          <div className="relative w-3" style={{ height: `${BAR_HEIGHT}px` }}>
            {tmin !== null && tmax !== null && (
              <div
                className={cn(
                  "absolute left-0 right-0 rounded-full",
                  tempBarColour(tmax),
                )}
                style={{
                  bottom: `${barBottom}px`,
                  height: `${Math.max(2, barTop - barBottom)}px`,
                }}
                aria-hidden
              />
            )}
            {meanY !== null && (
              <div
                className="absolute -left-1 right-[-4px] h-0.5 bg-foreground/40"
                style={{ bottom: `${meanY}px` }}
                aria-hidden
              />
            )}
          </div>

          <div className="text-[11px] tabular-nums">
            {tmax !== null ? `${tmax.toFixed(0)}°` : "—"}
          </div>

          {/* Precip bar */}
          <div className="relative h-7 w-3">
            {(day.precip_mm ?? 0) > 0 && (
              <div
                className="absolute bottom-0 left-0 right-0 rounded-sm bg-blue-500"
                style={{ height: `${Math.max(2, precipPx)}px` }}
              />
            )}
          </div>
          <div className="text-[10px] tabular-nums text-blue-600">
            {(day.precip_mm ?? 0) > 0
              ? `${(day.precip_mm ?? 0).toFixed(1)}`
              : "—"}
          </div>
        </div>
      </TooltipTrigger>
      <TooltipContent side="top" align="center" className="w-56">
        <DayDetail day={day} />
      </TooltipContent>
    </Tooltip>
  );
}

function DayDetail({ day }: { day: WeatherDay }) {
  const d = new Date(day.observed_on);
  return (
    <div className="space-y-1.5">
      <div className="font-semibold">
        {format(d, "EEEE, d MMMM", { locale: uk })}
      </div>
      <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[11px]">
        <Item label="Темп. мін / макс">
          {fmtN(day.temp_min_c, 0)}° / {fmtN(day.temp_max_c, 0)}°
        </Item>
        <Item label="Середня">{fmtN(day.temp_mean_c, 0)}°</Item>
        <Item label="Опади">
          {(day.precip_mm ?? 0).toFixed(1)} мм
        </Item>
        <Item label="Вологість">{fmtN(day.humidity_pct, 0)}%</Item>
        <Item label="Вітер">
          {day.wind_speed_max_ms !== null
            ? `${day.wind_speed_max_ms.toFixed(1)} м/с`
            : "—"}
        </Item>
        <Item label="Хмарність">
          {day.cloud_cover_pct !== null
            ? `${day.cloud_cover_pct.toFixed(0)}%`
            : "—"}
        </Item>
        <Item label="Радіація">
          {day.radiation_mj !== null
            ? `${day.radiation_mj.toFixed(1)} МДж`
            : "—"}
        </Item>
        <Item label="Волога ґрунту">
          {day.soil_moisture_0_10cm !== null
            ? `${(day.soil_moisture_0_10cm * 100).toFixed(0)}%`
            : "—"}
        </Item>
      </div>
    </div>
  );
}

function Item({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium tabular-nums">{children}</span>
    </div>
  );
}

function fmtN(v: number | null, digits: number): string {
  return v !== null ? v.toFixed(digits) : "—";
}


// ─── Sky icon picker ───────────────────────────────────────


function pickSkyIcon(d: WeatherDay) {
  const precip = d.precip_mm ?? 0;
  const cloud = d.cloud_cover_pct;
  if (precip > 5) return CloudRain;
  if (cloud === null) {
    // Fallback when cloud_cover wasn't fetched: use radiation as proxy.
    if (d.radiation_mj !== null && d.radiation_mj > 18) return Sun;
    return Cloud;
  }
  if (cloud < 25) return Sun;
  if (cloud < 60) return CloudSun;
  if (cloud < 85) return Cloudy;
  return Cloud;
}

function skyColour(d: WeatherDay): string {
  const precip = d.precip_mm ?? 0;
  if (precip > 5) return "text-blue-500";
  const cloud = d.cloud_cover_pct;
  if (cloud !== null && cloud < 25) return "text-amber-500";
  if (cloud !== null && cloud > 80) return "text-slate-500";
  return "text-muted-foreground";
}

function tempBarColour(tmax: number): string {
  if (tmax >= 30) return "bg-red-500";
  if (tmax >= 25) return "bg-orange-400";
  if (tmax >= 15) return "bg-yellow-400";
  if (tmax >= 5) return "bg-emerald-400";
  if (tmax >= -5) return "bg-sky-400";
  return "bg-sky-600";
}


// ─── Unused-import keep-alive (silences TS without runtime cost) ──


void Droplets;
void Wind;
void Sprout;
