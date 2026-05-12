import { useMemo } from "react";

import type { DashboardFieldRow } from "@/api/dashboard";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface Props {
  fields: DashboardFieldRow[];
}

/**
 * Side-by-side: your field's current NDVI vs the oblast-aggregated mean
 * derived from the Week 6 Sentinel-2 dataset. Only rows that have BOTH
 * values appear — rest are filtered out so the card stays informative.
 */
export function OblastComparisonTable({ fields }: Props) {
  const rows = useMemo(() => {
    return fields
      .filter((f) => f.current_ndvi !== null && f.oblast_avg_ndvi !== null)
      .map((f) => ({
        ...f,
        delta: (f.current_ndvi as number) - (f.oblast_avg_ndvi as number),
      }))
      .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
  }, [fields]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">
          Поле vs середнє по області
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        {rows.length === 0 ? (
          <div className="px-6 py-6 text-sm text-muted-foreground">
            Дані рівня області ще не зібрані. Запустіть Week 6 pipeline
            (<code>scripts/collect_oblast_s2.py</code> + <code>build_features_v2.py</code>)
            щоб увімкнути порівняння.
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left text-xs text-muted-foreground">
                <th className="px-4 py-2 font-medium">Поле</th>
                <th className="px-2 py-2 text-right font-medium">Ваш NDVI</th>
                <th className="px-2 py-2 text-right font-medium">Область</th>
                <th className="px-2 py-2 text-right font-medium">Δ</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr
                  key={r.field_id}
                  className="border-b last:border-b-0 hover:bg-accent/30"
                >
                  <td className="px-4 py-2 font-medium">{r.name}</td>
                  <td className="px-2 py-2 text-right tabular-nums">
                    {(r.current_ndvi as number).toFixed(2)}
                  </td>
                  <td className="px-2 py-2 text-right tabular-nums text-muted-foreground">
                    {(r.oblast_avg_ndvi as number).toFixed(2)}
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
        )}
      </CardContent>
    </Card>
  );
}
