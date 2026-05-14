import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { Info, MapPin } from "lucide-react";

import type { DashboardFieldRow } from "@/api/dashboard";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface Props {
  fields: DashboardFieldRow[];
}

interface ComparisonRow {
  field_id: number;
  name: string;
  crop_type: string;
  predicted_tha: number;
  oblast_avg_yield_tha: number;
  oblast_name: string | null;
  baseline_year: number | null;
  delta_pct: number;
  model_r2: number | null;
  significance: "within_noise" | "significant" | "unknown";
  grey_band_pct: number;
}


/**
 * Threshold that splits "within model uncertainty" from "significant
 * deviation" for the |Δ%| value vs oblast baseline.
 *
 * Empirically: at R²=0.5 a ~20% gap is roughly within the model's
 * own error band; at R²=0.8 the band tightens to ~10%. The (1-R²)×40
 * formula gives a single knob that scales the grey-out radius with
 * the model's actual test accuracy for that crop. Weak crops get
 * forgiveness, strong crops get sharper signal.
 */
function greyBandPct(r2: number | null): number {
  if (r2 === null || r2 <= 0) return 25; // unknown / negative R² → default 25%
  return Math.max(5, (1 - r2) * 40);
}


function gapSignificance(
  r2: number | null,
  delta_pct: number,
): "within_noise" | "significant" | "unknown" {
  if (r2 === null) return "unknown";
  return Math.abs(delta_pct) < greyBandPct(r2) ? "within_noise" : "significant";
}

/**
 * Side-by-side: this field's predicted yield (from the ML model) vs
 * the published Держстат mean yield for its (oblast, crop). Three
 * things at once — your forecast, the regional reality, and the gap
 * in % terms.
 *
 * v7-era rewrite — the previous version compared NDVI peaks (less
 * meaningful for an end-user agronomist who thinks in t/ha) and was
 * tied to the legacy v2 parquet's 8-oblast / 3-crop coverage. We now
 * read the v3 parquet via `oblast_avg_yield()` on the backend (24
 * oblasts × 13 crops, `is_real_yield=True` filter) so virtually every
 * field falls inside coverage.
 *
 * Empty-state distinguishes three failure modes:
 *   - no fields at all,
 *   - centroid doesn't land in any known oblast (geojson gap / outside
 *     Ukraine),
 *   - oblast known but no Держстат row for that crop + oblast combo
 *     (e.g. a niche crop in a small-area oblast that wasn't in the
 *     2018-2021 published bulletins).
 */
export function OblastComparisonTable({ fields }: Props) {
  const navigate = useNavigate();
  const { t } = useTranslation();
  const rows = useMemo<ComparisonRow[]>(() => {
    return fields
      .filter(
        (f): f is DashboardFieldRow & {
          predicted_tha: number;
          oblast_avg_yield_tha: number;
        } =>
          f.predicted_tha !== null && f.oblast_avg_yield_tha !== null,
      )
      .map((f) => {
        // Defensive: denominator can theoretically be 0 / very small —
        // clamp so we never produce Infinity or absurd % values.
        const denom = f.oblast_avg_yield_tha > 0 ? f.oblast_avg_yield_tha : 1;
        const delta_pct = ((f.predicted_tha - f.oblast_avg_yield_tha) / denom) * 100;
        const grey_band_pct = greyBandPct(f.model_r2_for_crop);
        return {
          field_id: f.field_id,
          name: f.name,
          crop_type: f.crop_type,
          predicted_tha: f.predicted_tha,
          oblast_avg_yield_tha: f.oblast_avg_yield_tha,
          oblast_name: f.oblast_name,
          baseline_year: f.oblast_avg_yield_year,
          delta_pct,
          model_r2: f.model_r2_for_crop,
          significance: gapSignificance(f.model_r2_for_crop, delta_pct),
          grey_band_pct,
        };
      })
      .sort((a, b) => Math.abs(b.delta_pct) - Math.abs(a.delta_pct));
  }, [fields]);

  // Categorise fields without a comparison so we can write a useful
  // empty state. "fieldsWithOblast" means we resolved a centroid →
  // oblast; "fieldsWithPrediction" means the predict-job has produced
  // a number we can compare against.
  const fieldsWithOblast = fields.filter((f) => f.oblast_name !== null);
  const fieldsWithPrediction = fields.filter((f) => f.predicted_tha !== null);
  const baselineYear =
    rows.find((r) => r.baseline_year !== null)?.baseline_year ?? null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Поле vs середнє по області</CardTitle>
        {baselineYear !== null && (
          <p className="text-xs text-muted-foreground">
            База порівняння — фактичні Держстат-урожаї {baselineYear} р.
            {/* When the baseline is more than 1 year behind today's
                year, surface the staleness so users don't read it as
                "this year's yield" — happens for late-harvest crops
                (sugar_beet / corn / potato) whose November bulletin
                drops after the dashboard already showed a comparison
                against the prior year. */}
            {baselineYear < new Date().getFullYear() - 1 && (
              <span className="italic"> (найсвіжіший доступний)</span>
            )}
          </p>
        )}
      </CardHeader>
      <CardContent className="p-0">
        {rows.length === 0 ? (
          <EmptyState
            fieldsTotal={fields.length}
            fieldsWithKnownOblast={fieldsWithOblast.length}
            fieldsWithPrediction={fieldsWithPrediction.length}
          />
        ) : (
          <>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-left text-xs text-muted-foreground">
                  <th className="px-4 py-2 font-medium">Поле</th>
                  <th className="px-2 py-2 font-medium">Область</th>
                  <th className="px-2 py-2 text-right font-medium">
                    Прогноз (т/га)
                  </th>
                  <th className="px-2 py-2 text-right font-medium">
                    Область (т/га)
                  </th>
                  <th className="px-2 py-2 text-right font-medium">Δ</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr
                    key={r.field_id}
                    className="cursor-pointer border-b last:border-b-0 hover:bg-accent/30"
                    onClick={() => navigate(`/fields?selected=${r.field_id}`)}
                  >
                    <td className="px-4 py-2 font-medium">{r.name}</td>
                    <td className="px-2 py-2 text-xs text-muted-foreground">
                      <span className="inline-flex items-center gap-1">
                        <MapPin className="h-3 w-3" />
                        {r.oblast_name ?? "—"}
                      </span>
                    </td>
                    <td className="px-2 py-2 text-right tabular-nums">
                      {r.predicted_tha.toFixed(1)}
                    </td>
                    <td className="px-2 py-2 text-right tabular-nums text-muted-foreground">
                      {r.oblast_avg_yield_tha.toFixed(1)}
                    </td>
                    <td className="px-2 py-2 text-right">
                      <DeltaCell row={r} t={t} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="border-t bg-muted/30 px-4 py-2 text-[11px] text-muted-foreground">
              Сірим кольором — відхилення в межах похибки моделі для
              культури (наведіть курсор на «ⓘ»). База точності — test
              2021 (R² на реальних Держстат-урожаях).
            </p>
            {fields.length > rows.length && (
              <p className="border-t bg-muted/30 px-4 py-2 text-[11px] text-muted-foreground">
                Ще {fields.length - rows.length}{" "}
                {plural(
                  fields.length - rows.length,
                  "поле",
                  "поля",
                  "полів",
                )}{" "}
                не порівнюються: ще немає прогнозу або немає
                Держстат-даних за цю культуру та область.
              </p>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

/**
 * Renders the Δ% cell — coloured red/green for significant gaps,
 * greyed out with a tooltip for gaps within the model's own error
 * band. The tooltip explains the calibration: "model for this crop
 * explains ~N% of variance; deviations under ~M% are within noise".
 */
function DeltaCell({
  row,
  t,
}: {
  row: ComparisonRow;
  t: (key: string) => string;
}) {
  const sign = row.delta_pct >= 0 ? "+" : "";
  const formatted = `${sign}${row.delta_pct.toFixed(1)}%`;

  // Within-noise band: grey, info-icon, tooltip with model accuracy.
  if (row.significance === "within_noise") {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <span className="inline-flex items-center gap-1 tabular-nums font-medium text-muted-foreground">
            {formatted}
            <Info className="h-3 w-3 opacity-60" />
          </span>
        </TooltipTrigger>
        <TooltipContent side="left" className="max-w-[280px] text-xs">
          У межах похибки моделі.{" "}
          {row.model_r2 !== null && (
            <>
              Для культури «{t(`fields.crops.${row.crop_type}`)}» модель
              пояснює близько {(row.model_r2 * 100).toFixed(0)}% варіації
              врожайності, тому відхилення менше ~
              {row.grey_band_pct.toFixed(0)}% від обласного середнього є
              нормальним коливанням.
            </>
          )}
        </TooltipContent>
      </Tooltip>
    );
  }

  // Significant gap: red/green + tooltip explaining what to check.
  const colour =
    row.delta_pct >= 0 ? "text-emerald-600" : "text-red-600";
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          className={`inline-flex items-center gap-1 tabular-nums font-medium ${colour}`}
        >
          {formatted}
          <Info className="h-3 w-3 opacity-50" />
        </span>
      </TooltipTrigger>
      <TooltipContent side="left" className="max-w-[280px] text-xs">
        Значне відхилення від обласного середнього.{" "}
        {row.model_r2 !== null && (
          <>
            Модель для культури «{t(`fields.crops.${row.crop_type}`)}»
            пояснює близько {(row.model_r2 * 100).toFixed(0)}% варіації —
            різниця понад ~{row.grey_band_pct.toFixed(0)}% виходить за
            межі типової похибки і варта перевірки (NDVI поточного
            сезону, ґрунтові умови, історія сівозміни).
          </>
        )}
      </TooltipContent>
    </Tooltip>
  );
}


function EmptyState({
  fieldsTotal,
  fieldsWithKnownOblast,
  fieldsWithPrediction,
}: {
  fieldsTotal: number;
  fieldsWithKnownOblast: number;
  fieldsWithPrediction: number;
}) {
  if (fieldsTotal === 0) {
    return (
      <div className="px-6 py-6 text-sm text-muted-foreground">
        У вас ще немає полів. Додайте поле на сторінці «Поля».
      </div>
    );
  }
  if (fieldsWithKnownOblast === 0) {
    return (
      <div className="px-6 py-6 text-sm text-muted-foreground">
        Не вдалося визначити область для жодного поля. Перевірте, що
        центроїди полів лежать у межах України та що{" "}
        <code className="rounded bg-muted px-1">
          backend/data/raw/ukraine_oblasts.geojson
        </code>{" "}
        доступний.
      </div>
    );
  }
  if (fieldsWithPrediction === 0) {
    return (
      <div className="px-6 py-6 text-sm text-muted-foreground">
        Жоден прогноз ще не порахований. Натисніть «Перерахувати» на
        картці прогнозу будь-якого поля, щоб запустити модель.
      </div>
    );
  }
  return (
    <div className="space-y-2 px-6 py-6 text-sm text-muted-foreground">
      <p>
        Немає даних Держстат за культуру/область для жодного з ваших
        полів. Якщо ви додали нішеву культуру або область, опубліковані
        бюлетені 2018-2021 її не покривають — спробуйте порівняти
        окремо за наявності власних даних.
      </p>
    </div>
  );
}

function plural(n: number, one: string, few: string, many: string): string {
  const mod10 = n % 10;
  const mod100 = n % 100;
  if (mod10 === 1 && mod100 !== 11) return one;
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return few;
  return many;
}
