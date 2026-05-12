import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { MapPin } from "lucide-react";

import type { DashboardFieldRow } from "@/api/dashboard";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface Props {
  fields: DashboardFieldRow[];
}

interface ComparisonRow {
  field_id: number;
  name: string;
  current_ndvi: number;
  oblast_avg_ndvi: number;
  oblast_name: string | null;
  baseline_year: number | null;
  delta: number;
}

/**
 * Side-by-side: your field's current NDVI vs the Week 6 oblast-aggregated
 * baseline (cropland-mask sampled Sentinel-2). The card teaches three
 * things at once — your value, the regional reference, and the gap.
 *
 * Empty-state distinguishes the two reasons a field might be missing:
 *   - we don't yet know the oblast (centroid outside Ukraine / geojson
 *     unavailable) — typically a setup issue;
 *   - we know the oblast but the Week 6 dataset doesn't cover it yet —
 *     a content gap that fills as collect_oblast_s2.py advances.
 */
export function OblastComparisonTable({ fields }: Props) {
  const navigate = useNavigate();
  const rows = useMemo<ComparisonRow[]>(() => {
    return fields
      .filter(
        (f): f is DashboardFieldRow & { current_ndvi: number; oblast_avg_ndvi: number } =>
          f.current_ndvi !== null && f.oblast_avg_ndvi !== null,
      )
      .map((f) => ({
        field_id: f.field_id,
        name: f.name,
        current_ndvi: f.current_ndvi,
        oblast_avg_ndvi: f.oblast_avg_ndvi,
        oblast_name: f.oblast_name,
        baseline_year: f.oblast_baseline_year,
        delta: f.current_ndvi - f.oblast_avg_ndvi,
      }))
      .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
  }, [fields]);

  // Categorise fields without a comparison so we can write a useful empty
  // state ("no oblast" vs "oblast known but no data").
  const fieldsWithOblast = fields.filter((f) => f.oblast_name !== null);
  const baselineYear =
    rows.find((r) => r.baseline_year !== null)?.baseline_year ?? null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Поле vs середнє по області</CardTitle>
        {baselineYear !== null && (
          <p className="text-xs text-muted-foreground">
            Базова лінія за Sentinel-2 датасет {baselineYear} р.
          </p>
        )}
      </CardHeader>
      <CardContent className="p-0">
        {rows.length === 0 ? (
          <EmptyState
            fieldsTotal={fields.length}
            fieldsWithKnownOblast={fieldsWithOblast.length}
          />
        ) : (
          <>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-left text-xs text-muted-foreground">
                  <th className="px-4 py-2 font-medium">Поле</th>
                  <th className="px-2 py-2 font-medium">Область</th>
                  <th className="px-2 py-2 text-right font-medium">Ваш NDVI</th>
                  <th className="px-2 py-2 text-right font-medium">Область</th>
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
                      {r.current_ndvi.toFixed(2)}
                    </td>
                    <td className="px-2 py-2 text-right tabular-nums text-muted-foreground">
                      {r.oblast_avg_ndvi.toFixed(2)}
                    </td>
                    <td
                      className={`px-2 py-2 text-right tabular-nums font-medium ${
                        r.delta > 0 ? "text-emerald-600" : "text-red-600"
                      }`}
                    >
                      {r.delta > 0 ? "+" : ""}
                      {r.delta.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {fields.length > rows.length && (
              <p className="border-t bg-muted/30 px-4 py-2 text-[11px] text-muted-foreground">
                Ще {fields.length - rows.length} {plural(fields.length - rows.length, "поле", "поля", "полів")}
                {" "}не порівнюються: або їх область поза покриттям Week 6
                датасета, або поточний NDVI ще не зібрано.
              </p>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

function EmptyState({
  fieldsTotal,
  fieldsWithKnownOblast,
}: {
  fieldsTotal: number;
  fieldsWithKnownOblast: number;
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
        центроїди полів лежать у межах України та що
        {" "}<code className="rounded bg-muted px-1">backend/data/raw/ukraine_oblasts.geojson</code>
        {" "}доступний (генерується скриптом{" "}
        <code className="rounded bg-muted px-1">scripts/download_oblast_geometries.py</code>).
      </div>
    );
  }
  return (
    <div className="space-y-2 px-6 py-6 text-sm text-muted-foreground">
      <p>
        Області ваших полів є <strong>поза покриттям</strong> поточного
        Sentinel-2 датасета.
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
