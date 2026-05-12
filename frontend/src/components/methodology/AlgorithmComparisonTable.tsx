import type { ModelInfo } from "@/api/methodology";
import { Card } from "@/components/ui/card";

const CROP_LABELS: Record<string, string> = {
  wheat: "Пшениця",
  corn: "Кукурудза",
  sunflower: "Соняшник",
};

const FAMILY_LABELS: Record<string, string> = {
  xgboost: "XGBoost",
  rf: "Random Forest",
  lstm: "LSTM",
};

interface Props {
  models: ModelInfo[];
}

/**
 * 3 × 3 grid: rows = crop, cols = algorithm family. Each cell shows the
 * latest version's test R² + MAE for that combo. Empty cells mean the
 * model file isn't trained yet — a normal state while the magistr pipeline
 * is being executed.
 */
export function AlgorithmComparisonTable({ models }: Props) {
  const crops = ["wheat", "corn", "sunflower"] as const;
  const families = ["rf", "xgboost", "lstm"] as const;

  const lookup: Record<string, ModelInfo | undefined> = {};
  for (const m of models) {
    const key = `${m.crop}_${m.family}`;
    const existing = lookup[key];
    // Pick the highest version per (crop, family).
    if (!existing || existing.version < m.version) lookup[key] = m;
  }

  return (
    <Card className="overflow-x-auto p-0">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b text-xs text-muted-foreground">
            <th className="px-3 py-2 text-left font-medium">Культура</th>
            {families.map((f) => (
              <th key={f} className="px-3 py-2 text-left font-medium">
                {FAMILY_LABELS[f]}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {crops.map((crop) => (
            <tr key={crop} className="border-b last:border-b-0">
              <td className="px-3 py-2 font-medium">{CROP_LABELS[crop]}</td>
              {families.map((family) => (
                <td key={family} className="px-3 py-2">
                  <Cell model={lookup[`${crop}_${family}`]} />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}

function Cell({ model }: { model: ModelInfo | undefined }) {
  if (!model) {
    return <span className="text-xs text-muted-foreground">—</span>;
  }
  const test = model.metrics?.test;
  if (!test) {
    return (
      <div className="text-xs">
        <div>{model.version}</div>
        <div className="text-muted-foreground">метрики недоступні</div>
      </div>
    );
  }
  const r2Color =
    test.r2 > 0.4 ? "text-emerald-600" : test.r2 > 0 ? "text-amber-600" : "text-red-600";
  return (
    <div className="text-xs leading-tight">
      <div className="font-mono text-[10px] text-muted-foreground">{model.version}</div>
      <div>
        R² <span className={`font-semibold ${r2Color}`}>{test.r2.toFixed(2)}</span>
      </div>
      <div className="text-muted-foreground">
        MAE {test.mae.toFixed(2)} т/га
      </div>
    </div>
  );
}
