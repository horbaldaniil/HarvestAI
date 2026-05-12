import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";
import { ArrowDownRight, ArrowUpRight } from "lucide-react";

import type { FieldYoYDelta } from "@/api/dashboard";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { CROP_COLORS } from "@/lib/colors";
import { cn } from "@/lib/utils";

interface Props {
  movers: FieldYoYDelta[];
  currentYear: number;
}

/**
 * Replaces the old portfolio-average YoY widget with the three fields whose
 * NDVI moved most year-over-year — positive or negative. Clearer because:
 *  - every row names the actual field (not "average across all"),
 *  - signed diff makes wins and regressions obviously distinct,
 *  - the year labels make the comparison unambiguous,
 *  - clicking drills into the field for the full story.
 */
export function TopMoversCard({ movers, currentYear }: Props) {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const prevYear = currentYear - 1;

  if (movers.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Найбільші зміни NDVI</CardTitle>
        </CardHeader>
        <CardContent className="text-xs text-muted-foreground">
          Недостатньо даних для порівняння (потрібен ≥ 1 рік історії NDVI на полях).
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Найбільші зміни NDVI</CardTitle>
        <CardDescription>
          Середній NDVI {prevYear} → {currentYear}, поля з найбільшою динамікою
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-2">
        {movers.map((m) => (
          <button
            key={m.field_id}
            type="button"
            onClick={() => navigate(`/fields?selected=${m.field_id}`)}
            className="flex w-full items-center gap-3 rounded-md border bg-card p-2 text-left transition-colors hover:bg-accent/40"
          >
            <span
              className="h-2.5 w-2.5 rounded-sm shrink-0"
              style={{ background: CROP_COLORS[m.crop_type] }}
              aria-hidden
            />
            <div className="min-w-0 flex-1">
              <div className="truncate text-sm font-medium">{m.name}</div>
              <div className="mt-0.5 text-[10px] text-muted-foreground tabular-nums">
                <span>
                  {prevYear}: <span className="font-medium">{m.prev_year_ndvi.toFixed(2)}</span>
                </span>
                <span className="mx-1.5">→</span>
                <span>
                  {currentYear}: <span className="font-medium">{m.current_year_ndvi.toFixed(2)}</span>
                </span>
                <span className="ml-2 text-muted-foreground/70">
                  · {t(`fields.crops.${m.crop_type}`)}
                </span>
              </div>
            </div>
            <DiffPill pct={m.diff_pct} />
          </button>
        ))}
      </CardContent>
    </Card>
  );
}

function DiffPill({ pct }: { pct: number }) {
  const positive = pct >= 0;
  const Icon = positive ? ArrowUpRight : ArrowDownRight;
  const colour = positive
    ? "text-emerald-700 bg-emerald-50 border-emerald-200"
    : "text-red-700 bg-red-50 border-red-200";
  return (
    <span
      className={cn(
        "inline-flex items-center gap-0.5 rounded border px-1.5 py-0.5 text-xs font-medium tabular-nums",
        colour,
      )}
    >
      <Icon className="h-3 w-3" />
      {positive ? "+" : ""}
      {pct.toFixed(1)}%
    </span>
  );
}
