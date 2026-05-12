import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  CloudRain,
  Droplets,
  Loader2,
  MapPin,
  Sun,
  Thermometer,
  Wind,
} from "lucide-react";

import { AppShell } from "@/components/layout/AppShell";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { WeatherAdvicesPanel } from "@/components/weather/WeatherAdvicesPanel";
import { WeatherDayStrip } from "@/components/weather/WeatherDayStrip";
import { useFields } from "@/hooks/useFields";
import { useWeatherDetail } from "@/hooks/useWeather";
import { CROP_COLORS } from "@/lib/colors";

export function WeatherPage() {
  const { t } = useTranslation();
  const { data: fields, isLoading: fieldsLoading } = useFields();
  const [selectedFieldId, setSelectedFieldId] = useState<number | null>(null);

  // Auto-select the first field once they load.
  useEffect(() => {
    if (selectedFieldId === null && fields && fields.length > 0) {
      setSelectedFieldId(fields[0].id);
    }
  }, [fields, selectedFieldId]);

  const { data, isLoading: weatherLoading } = useWeatherDetail(
    selectedFieldId,
    14,
  );

  if (fieldsLoading) {
    return (
      <AppShell>
        <div className="flex h-64 items-center justify-center">
          <Loader2 className="h-6 w-6 animate-spin text-primary" />
        </div>
      </AppShell>
    );
  }

  if (!fields || fields.length === 0) {
    return (
      <AppShell>
        <header className="mb-6">
          <h1 className="text-3xl font-bold tracking-tight">Погода</h1>
        </header>
        <Card>
          <CardContent className="py-10 text-center text-sm text-muted-foreground">
            Спершу додайте поле на сторінці «Поля» — після цього фоновий
            процес завантажить прогноз із Open-Meteo.
          </CardContent>
        </Card>
      </AppShell>
    );
  }

  const sevenDay = (data?.days ?? []).slice(0, 7);
  const fourteenDay = data?.days ?? [];

  return (
    <AppShell>
      <div className="space-y-6">
        <header className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Погода</h1>
            <p className="text-muted-foreground">
              Детальний прогноз з рекомендаціями для одного поля.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-sm text-muted-foreground">Поле:</span>
            <Select
              value={selectedFieldId !== null ? String(selectedFieldId) : ""}
              onValueChange={(v) => setSelectedFieldId(Number(v))}
            >
              <SelectTrigger className="min-w-[14rem]">
                <SelectValue placeholder="Виберіть поле" />
              </SelectTrigger>
              <SelectContent>
                {fields.map((f) => (
                  <SelectItem key={f.id} value={String(f.id)}>
                    <span className="flex items-center gap-2">
                      <span
                        className="h-2 w-2 rounded-sm"
                        style={{ background: CROP_COLORS[f.crop_type] }}
                        aria-hidden
                      />
                      <span>{f.name}</span>
                    </span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </header>

        {weatherLoading || !data ? (
          <div className="flex h-64 items-center justify-center">
            <Loader2 className="h-6 w-6 animate-spin text-primary" />
          </div>
        ) : (
          <>
            <FieldContextCard data={data} t={t} />
            <SevenDayKpis days={sevenDay} />

            <Card>
              <CardHeader>
                <CardTitle className="text-base">
                  Прогноз на 14 днів
                </CardTitle>
              </CardHeader>
              <CardContent>
                <WeatherDayStrip days={fourteenDay} />
                <div className="mt-3 flex flex-wrap items-center gap-3 text-[10px] text-muted-foreground">
                  <span className="flex items-center gap-1">
                    <span className="h-2 w-3 rounded-sm bg-red-500" /> &gt;30°C
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="h-2 w-3 rounded-sm bg-orange-400" /> 25–30°C
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="h-2 w-3 rounded-sm bg-yellow-400" /> 15–25°C
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="h-2 w-3 rounded-sm bg-emerald-400" /> 5–15°C
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="h-2 w-3 rounded-sm bg-sky-400" /> -5–5°C
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="h-2 w-3 rounded-sm bg-blue-500" /> Опади (мм)
                  </span>
                </div>
              </CardContent>
            </Card>

            <WeatherAdvicesPanel advices={data.advices} />
          </>
        )}
      </div>
    </AppShell>
  );
}


// ─── Context strip — field + location ──────────────────────────


function FieldContextCard({
  data,
  t,
}: {
  data: { field_name: string; crop_type: string; centroid_lat: number | null; centroid_lon: number | null };
  t: (key: string) => string;
}) {
  return (
    <Card>
      <CardContent className="flex flex-wrap items-center gap-x-6 gap-y-2 py-3 text-sm">
        <div>
          <span className="text-xs text-muted-foreground">Поле:</span>{" "}
          <span className="font-medium">{data.field_name}</span>
        </div>
        <div>
          <span className="text-xs text-muted-foreground">Культура:</span>{" "}
          <span className="font-medium">
            {t(`fields.crops.${data.crop_type}`)}
          </span>
        </div>
        {data.centroid_lat !== null && (
          <div className="flex items-center gap-1 text-muted-foreground">
            <MapPin className="h-3.5 w-3.5" />
            <span className="text-xs tabular-nums">
              {data.centroid_lat.toFixed(3)}°, {data.centroid_lon?.toFixed(3)}°
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}


// ─── 7-day KPI row (averaged over the next 7 days) ─────────────


function SevenDayKpis({
  days,
}: {
  days: Array<{
    temp_max_c: number | null;
    temp_min_c: number | null;
    precip_mm: number | null;
    wind_speed_max_ms: number | null;
    cloud_cover_pct: number | null;
    soil_moisture_0_10cm: number | null;
  }>;
}) {
  const tempMax7 = max(days.map((d) => d.temp_max_c));
  const tempMin7 = min(days.map((d) => d.temp_min_c));
  const precip7 = sum(days.map((d) => d.precip_mm));
  const windMax7 = max(days.map((d) => d.wind_speed_max_ms));
  const cloud7 = avg(days.map((d) => d.cloud_cover_pct));
  const soil7 = avg(days.map((d) => d.soil_moisture_0_10cm));
  const heatDays = days.filter(
    (d) => d.temp_max_c !== null && d.temp_max_c > 30,
  ).length;

  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      <Kpi
        icon={<Thermometer className="h-4 w-4 text-red-500" />}
        label="Темп. макс / мін"
        value={
          tempMax7 !== null && tempMin7 !== null
            ? `${tempMax7.toFixed(0)}° / ${tempMin7.toFixed(0)}°`
            : "—"
        }
      />
      <Kpi
        icon={<CloudRain className="h-4 w-4 text-blue-500" />}
        label="Опади (сума)"
        value={precip7 !== null ? `${precip7.toFixed(1)} мм` : "—"}
      />
      <Kpi
        icon={<Wind className="h-4 w-4 text-slate-500" />}
        label="Макс. вітер"
        value={windMax7 !== null ? `${windMax7.toFixed(1)} м/с` : "—"}
      />
      <Kpi
        icon={<Sun className="h-4 w-4 text-amber-500" />}
        label="Гарячі дні"
        value={`${heatDays} дн.`}
      />
      <Kpi
        icon={<Sun className="h-4 w-4 text-orange-500" />}
        label="Хмарність ср."
        value={cloud7 !== null ? `${cloud7.toFixed(0)}%` : "—"}
      />
      <Kpi
        icon={<Droplets className="h-4 w-4 text-emerald-600" />}
        label="Волога ґрунту"
        value={soil7 !== null ? `${(soil7 * 100).toFixed(0)}%` : "—"}
      />
    </div>
  );
}

function Kpi({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <Card>
      <CardContent className="flex items-center gap-3 py-3">
        {icon}
        <div className="min-w-0">
          <div className="text-[11px] text-muted-foreground">{label}</div>
          <div className="text-lg font-semibold tabular-nums">{value}</div>
        </div>
      </CardContent>
    </Card>
  );
}

function max(arr: Array<number | null>): number | null {
  const xs = arr.filter((v): v is number => v !== null);
  return xs.length ? Math.max(...xs) : null;
}
function min(arr: Array<number | null>): number | null {
  const xs = arr.filter((v): v is number => v !== null);
  return xs.length ? Math.min(...xs) : null;
}
function sum(arr: Array<number | null>): number | null {
  const xs = arr.filter((v): v is number => v !== null);
  return xs.length ? xs.reduce((a, b) => a + b, 0) : null;
}
function avg(arr: Array<number | null>): number | null {
  const s = sum(arr);
  const xs = arr.filter((v): v is number => v !== null);
  return s !== null && xs.length > 0 ? s / xs.length : null;
}
