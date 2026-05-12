import { useMemo } from "react";
import { useTranslation } from "react-i18next";

import type { CropBreakdownItem, CropType } from "@/api/dashboard";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CROP_COLORS } from "@/lib/colors";

interface Props {
  breakdown: CropBreakdownItem[];
}

/**
 * Gantt-style phenology timeline: one row per crop the user actually has,
 * with the 12 calendar months as columns. Each phase block is coloured by
 * crop, lightened/intensified by current vs. past. The current month is
 * highlighted with a vertical line so the user sees "where we are now".
 *
 * Phenology dates mirror the static lookup in
 * `app/services/dashboard_analytics.py` and `app/data_reference/
 * crop_calendar.py` — keep all three in sync if you change one.
 * Months are 1-indexed (Jan=1 … Dec=12). Winter crops express the
 * "Спокій" phase as a wraparound range (start > end means it crosses
 * the year boundary — see CropRow rendering).
 */

const PHENOLOGY: Record<CropType, Array<{ phase: string; start: number; end: number }>> = {
  // — Original three (unchanged)
  wheat: [
    { phase: "Посів", start: 9, end: 10 },
    { phase: "Спокій", start: 11, end: 3 },
    { phase: "Ріст", start: 4, end: 5 },
    { phase: "Цвітіння", start: 6, end: 6 },
    { phase: "Дозрівання", start: 7, end: 7 },
    { phase: "Збір", start: 8, end: 8 },
  ],
  corn: [
    { phase: "Посів", start: 4, end: 4 },
    { phase: "Ріст", start: 5, end: 6 },
    { phase: "Цвітіння", start: 7, end: 7 },
    { phase: "Дозрівання", start: 8, end: 8 },
    { phase: "Збір", start: 9, end: 10 },
  ],
  sunflower: [
    { phase: "Посів", start: 4, end: 4 },
    { phase: "Ріст", start: 5, end: 6 },
    { phase: "Цвітіння", start: 7, end: 7 },
    { phase: "Дозрівання", start: 8, end: 8 },
    { phase: "Збір", start: 9, end: 9 },
  ],
  // — Oilseeds
  soybean: [
    { phase: "Посів", start: 5, end: 5 },
    { phase: "Ріст", start: 6, end: 6 },
    { phase: "Цвітіння", start: 7, end: 7 },
    { phase: "Дозрівання", start: 8, end: 8 },
    { phase: "Збір", start: 9, end: 9 },
  ],
  rapeseed: [
    // Winter — sown Sep, dormant Oct-Mar, flowers May
    { phase: "Посів", start: 9, end: 9 },
    { phase: "Спокій", start: 10, end: 3 },
    { phase: "Ріст", start: 4, end: 4 },
    { phase: "Цвітіння", start: 5, end: 5 },
    { phase: "Дозрівання", start: 6, end: 6 },
    { phase: "Збір", start: 7, end: 7 },
  ],
  // — Cereals
  barley: [
    { phase: "Посів", start: 4, end: 4 },
    { phase: "Ріст", start: 5, end: 5 },
    { phase: "Цвітіння", start: 6, end: 6 },
    { phase: "Збір", start: 7, end: 7 },
  ],
  rye: [
    // Winter — sown Oct
    { phase: "Посів", start: 10, end: 10 },
    { phase: "Спокій", start: 11, end: 3 },
    { phase: "Ріст", start: 4, end: 4 },
    { phase: "Цвітіння", start: 5, end: 5 },
    { phase: "Дозрівання", start: 6, end: 6 },
    { phase: "Збір", start: 7, end: 7 },
  ],
  oats: [
    { phase: "Посів", start: 4, end: 4 },
    { phase: "Ріст", start: 5, end: 5 },
    { phase: "Цвітіння", start: 6, end: 6 },
    { phase: "Збір", start: 7, end: 7 },
  ],
  buckwheat: [
    { phase: "Посів", start: 5, end: 5 },
    { phase: "Ріст", start: 6, end: 6 },
    { phase: "Цвітіння", start: 7, end: 7 },
    { phase: "Збір", start: 8, end: 8 },
  ],
  // — Legume
  peas: [
    { phase: "Посів", start: 4, end: 4 },
    { phase: "Ріст", start: 5, end: 5 },
    { phase: "Цвітіння", start: 6, end: 6 },
    { phase: "Збір", start: 7, end: 7 },
  ],
  // — Root crops
  sugar_beet: [
    { phase: "Посів", start: 4, end: 4 },
    { phase: "Ріст", start: 5, end: 7 },
    { phase: "Дозрівання", start: 8, end: 9 },
    { phase: "Збір", start: 10, end: 10 },
  ],
  potato: [
    { phase: "Посів", start: 4, end: 4 },
    { phase: "Ріст", start: 5, end: 6 },
    { phase: "Цвітіння", start: 7, end: 7 },
    { phase: "Дозрівання", start: 8, end: 8 },
    { phase: "Збір", start: 9, end: 9 },
  ],
  // — Forage
  corn_silage: [
    { phase: "Посів", start: 4, end: 4 },
    { phase: "Ріст", start: 5, end: 6 },
    { phase: "Цвітіння", start: 7, end: 7 },
    { phase: "Збір", start: 8, end: 8 },
  ],
};

const MONTH_LABELS = ["С", "Л", "Б", "К", "Т", "Ч", "Л", "С", "В", "Ж", "Л", "Г"];

export function CropCalendarTimeline({ breakdown }: Props) {
  const currentMonth = new Date().getMonth() + 1;
  const cropsWithFields = breakdown.filter((b) => b.field_count > 0);

  const cells = useMemo(() => {
    // Pre-compute the 12 boolean cells per crop: for each (crop, month),
    // does any phase touch it?  Then map to colour intensity.
    return cropsWithFields.map((b) => {
      const phases = PHENOLOGY[b.crop_type] ?? [];
      const grid = Array(12).fill(null) as (string | null)[];
      for (const ph of phases) {
        for (let m = 1; m <= 12; m++) {
          const hit =
            ph.start <= ph.end
              ? m >= ph.start && m <= ph.end
              : m >= ph.start || m <= ph.end;
          if (hit) grid[m - 1] = ph.phase;
        }
      }
      return { crop: b.crop_type, grid };
    });
  }, [cropsWithFields]);

  if (cropsWithFields.length === 0) {
    return null;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Фенологія культур</CardTitle>
      </CardHeader>
      <CardContent className="overflow-x-auto p-0">
        <div className="min-w-[640px] p-4">
          <div className="grid grid-cols-[120px_repeat(12,1fr)] gap-1 text-xs">
            <div></div>
            {MONTH_LABELS.map((m, idx) => (
              <div
                key={idx}
                className={`text-center ${
                  idx + 1 === currentMonth
                    ? "font-bold text-primary"
                    : "text-muted-foreground"
                }`}
              >
                {m}
              </div>
            ))}
            {cells.map(({ crop, grid }) => (
              <CropRow
                key={crop}
                crop={crop}
                grid={grid}
                currentMonth={currentMonth}
              />
            ))}
          </div>
          <div className="mt-3 text-[10px] text-muted-foreground">
            Поточний місяць позначено вертикальною лінією. Фази — за українськими
            агрономічними нормами.
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function CropRow({
  crop,
  grid,
  currentMonth,
}: {
  crop: CropType;
  grid: (string | null)[];
  currentMonth: number;
}) {
  const { t } = useTranslation();
  const colour = CROP_COLORS[crop];
  return (
    <>
      <div className="flex items-center font-medium">{t(`fields.crops.${crop}`)}</div>
      {grid.map((phase, idx) => (
        <div
          key={idx}
          className="relative h-6 rounded"
          style={{
            background: phase ? colour : "transparent",
            opacity: phase ? 0.7 : 0,
          }}
          title={phase ?? ""}
        >
          {idx + 1 === currentMonth && (
            <div className="absolute -inset-y-1 left-1/2 w-0.5 -translate-x-1/2 bg-primary" />
          )}
        </div>
      ))}
    </>
  );
}
