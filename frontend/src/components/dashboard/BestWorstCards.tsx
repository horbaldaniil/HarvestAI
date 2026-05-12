import { useNavigate } from "react-router-dom";
import { Award, AlertTriangle } from "lucide-react";

import type { BestWorstField } from "@/api/dashboard";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

interface Props {
  best: BestWorstField | null;
  worst: BestWorstField | null;
}

/**
 * Two highlight cards — "найкраще поле" by NDVI and "найбільш ризикове"
 * by risk_score. Both link to /fields?selected=N for drill-down.
 */
export function BestWorstCards({ best, worst }: Props) {
  return (
    <div className="grid gap-4 sm:grid-cols-2">
      <FieldHighlight
        kind="best"
        field={best}
        emptyLabel="Немає поля з NDVI"
      />
      <FieldHighlight
        kind="worst"
        field={worst}
        emptyLabel="Усе в нормі — критичних ризиків немає."
      />
    </div>
  );
}

function FieldHighlight({
  kind,
  field,
  emptyLabel,
}: {
  kind: "best" | "worst";
  field: BestWorstField | null;
  emptyLabel: string;
}) {
  const navigate = useNavigate();
  const isBest = kind === "best";
  const Icon = isBest ? Award : AlertTriangle;
  const accent = isBest ? "text-emerald-600" : "text-red-600";
  const title = isBest ? "Найкраще поле" : "Найбільший ризик";

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">
          {title}
        </CardTitle>
        <Icon className={`h-5 w-5 ${accent}`} />
      </CardHeader>
      <CardContent>
        {field ? (
          <>
            <div className="text-2xl font-bold truncate">{field.name}</div>
            <div className="mt-1 text-xs text-muted-foreground">
              {field.reason}
            </div>
            <div className="mt-2 flex items-baseline gap-3 text-xs">
              {field.current_ndvi !== null && (
                <span>
                  NDVI{" "}
                  <span className="tabular-nums font-medium">
                    {field.current_ndvi.toFixed(2)}
                  </span>
                </span>
              )}
              {field.predicted_tha !== null && (
                <span>
                  Прогноз{" "}
                  <span className="tabular-nums font-medium">
                    {field.predicted_tha.toFixed(1)} т/га
                  </span>
                </span>
              )}
              {!isBest && (
                <span>
                  Risk{" "}
                  <span className={`tabular-nums font-medium ${accent}`}>
                    {field.risk_score}
                  </span>
                </span>
              )}
            </div>
            <Button
              size="sm"
              variant="outline"
              onClick={() => navigate(`/fields?selected=${field.field_id}`)}
              className="mt-3 h-7 text-xs"
            >
              Відкрити поле
            </Button>
          </>
        ) : (
          <div className="py-2 text-xs text-muted-foreground">{emptyLabel}</div>
        )}
      </CardContent>
    </Card>
  );
}
