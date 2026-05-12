import { useMemo, useState } from "react";
import { ArrowDown, ArrowUp, ArrowUpDown, Loader2 } from "lucide-react";

import type { LeaderboardRow } from "@/api/methodology";
import { useLeaderboard } from "@/hooks/useMethodology";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

/**
 * Sortable cross-crop leaderboard for the Phase-4 evaluation.
 *
 * Replaces the v2 3×3 grid that became unreadable at 13 crops × 6
 * families = 78 cells. Pattern follows what real ML dashboards
 * (PapersWithCode, OpenML) do: one row per (crop, family), sortable
 * columns, default sort by test R² descending.
 *
 * Reads from /api/methodology/leaderboard. Headline metrics:
 *   - test R² / RMSE / MAE / MAPE
 *   - LOOCV R² (spatial generalisation)
 *   - RepeatedKFold R² mean ± σ (variance bound)
 */
type SortKey =
  | "crop"
  | "family"
  | "test_r2"
  | "test_rmse"
  | "test_mae"
  | "test_mape"
  | "loocv_r2"
  | "kfold_r2_mean";

const FAMILY_LABEL: Record<string, string> = {
  rf: "Random Forest",
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
  catboost: "CatBoost",
  stack: "Stacked Ensemble",
  lstm: "LSTM",
};

const CROP_LABEL_UK: Record<string, string> = {
  wheat: "Пшениця",
  corn: "Кукурудза",
  sunflower: "Соняшник",
  soybean: "Соя",
  rapeseed: "Ріпак",
  barley: "Ячмінь",
  rye: "Жито",
  oats: "Овес",
  buckwheat: "Гречка",
  peas: "Горох",
  sugar_beet: "Цукровий буряк",
  potato: "Картопля",
  corn_silage: "Кукурудза на силос",
};

export function AlgorithmLeaderboard() {
  const { data, isLoading, error } = useLeaderboard();
  const [sortKey, setSortKey] = useState<SortKey>("test_r2");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [cropFilter, setCropFilter] = useState<string>("");
  const [familyFilter, setFamilyFilter] = useState<string>("");

  const rows = data?.rows ?? [];

  const filtered = useMemo(() => {
    return rows.filter(
      (r) =>
        (!cropFilter || r.crop === cropFilter) &&
        (!familyFilter || r.family === familyFilter),
    );
  }, [rows, cropFilter, familyFilter]);

  const sorted = useMemo(() => {
    const cmp = (a: LeaderboardRow, b: LeaderboardRow): number => {
      const va = a[sortKey];
      const vb = b[sortKey];
      // null sentinel — push nulls last regardless of direction.
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === "number" && typeof vb === "number") {
        return sortDir === "desc" ? vb - va : va - vb;
      }
      return sortDir === "desc"
        ? String(vb).localeCompare(String(va))
        : String(va).localeCompare(String(vb));
    };
    return [...filtered].sort(cmp);
  }, [filtered, sortKey, sortDir]);

  const crops = useMemo(() => {
    return [...new Set(rows.map((r) => r.crop))].sort();
  }, [rows]);
  const families = useMemo(() => {
    return [...new Set(rows.map((r) => r.family))].sort();
  }, [rows]);

  if (isLoading) {
    return (
      <Card>
        <CardContent className="flex h-32 items-center justify-center">
          <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    );
  }
  if (error || !data) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-sm text-muted-foreground">
          Не вдалося завантажити лідерборд моделей.
        </CardContent>
      </Card>
    );
  }
  if (rows.length === 0) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-sm text-muted-foreground">
          Запустіть{" "}
          <code className="rounded bg-muted px-1">scripts/evaluate_models.py</code>{" "}
          щоб заповнити лідерборд.
        </CardContent>
      </Card>
    );
  }

  function flip(key: SortKey) {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      // Most metric columns are "higher = better" → start desc.
      // Error metrics (rmse, mae, mape) are "lower = better" → start asc.
      const lowerIsBetter = ["test_rmse", "test_mae", "test_mape"].includes(key);
      setSortDir(lowerIsBetter ? "asc" : "desc");
    }
  }

  return (
    <Card>
      <CardHeader className="space-y-2">
        <CardTitle className="text-base">
          Лідерборд моделей
          <span className="ml-2 text-xs font-normal text-muted-foreground">
            {sorted.length} з {rows.length} (crop × family) комбінацій
          </span>
        </CardTitle>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <FilterSelect
            label="Культура"
            value={cropFilter}
            onChange={setCropFilter}
            options={crops.map((c) => ({ value: c, label: CROP_LABEL_UK[c] ?? c }))}
          />
          <FilterSelect
            label="Модель"
            value={familyFilter}
            onChange={setFamilyFilter}
            options={families.map((f) => ({ value: f, label: FAMILY_LABEL[f] ?? f }))}
          />
          {(cropFilter || familyFilter) && (
            <button
              type="button"
              onClick={() => {
                setCropFilter("");
                setFamilyFilter("");
              }}
              className="rounded border bg-muted px-2 py-0.5 hover:bg-accent"
            >
              Скинути
            </button>
          )}
        </div>
      </CardHeader>
      <CardContent className="overflow-x-auto p-0">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left text-xs text-muted-foreground">
              <HeaderCell
                label="Культура"
                onClick={() => flip("crop")}
                active={sortKey === "crop"}
                dir={sortDir}
              />
              <HeaderCell
                label="Модель"
                onClick={() => flip("family")}
                active={sortKey === "family"}
                dir={sortDir}
              />
              <HeaderCell
                label="R²"
                title="Test R² (вище — краще)"
                onClick={() => flip("test_r2")}
                active={sortKey === "test_r2"}
                dir={sortDir}
                numeric
              />
              <HeaderCell
                label="RMSE"
                title="Root mean squared error на test (нижче — краще)"
                onClick={() => flip("test_rmse")}
                active={sortKey === "test_rmse"}
                dir={sortDir}
                numeric
              />
              <HeaderCell
                label="MAE"
                title="Mean absolute error на test (нижче — краще)"
                onClick={() => flip("test_mae")}
                active={sortKey === "test_mae"}
                dir={sortDir}
                numeric
              />
              <HeaderCell
                label="MAPE %"
                title="Mean abs percentage error (нижче — краще)"
                onClick={() => flip("test_mape")}
                active={sortKey === "test_mape"}
                dir={sortDir}
                numeric
              />
              <HeaderCell
                label="LOOCV R²"
                title="Leave-one-oblast-out R² (просторова генералізація)"
                onClick={() => flip("loocv_r2")}
                active={sortKey === "loocv_r2"}
                dir={sortDir}
                numeric
              />
              <HeaderCell
                label="K-fold R² ± σ"
                title="RepeatedKFold(5×3) mean R² + σ (статистична варіація)"
                onClick={() => flip("kfold_r2_mean")}
                active={sortKey === "kfold_r2_mean"}
                dir={sortDir}
                numeric
              />
              <th className="px-3 py-2 text-right font-medium" title="n train / n test">
                n
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((r, idx) => (
              <tr
                key={`${r.crop}-${r.family}`}
                className={cn(
                  "border-b last:border-b-0",
                  idx % 2 === 1 && "bg-muted/20",
                )}
              >
                <td className="px-3 py-2 font-medium">
                  {CROP_LABEL_UK[r.crop] ?? r.crop}
                </td>
                <td className="px-3 py-2">{FAMILY_LABEL[r.family] ?? r.family}</td>
                <NumCell
                  v={r.test_r2}
                  digits={3}
                  emphasis={r.test_r2 != null && r.test_r2 > 0.5}
                />
                <NumCell v={r.test_rmse} digits={3} />
                <NumCell v={r.test_mae} digits={3} />
                <NumCell v={r.test_mape} digits={1} suffix="%" />
                <NumCell v={r.loocv_r2} digits={3} />
                <KfoldCell mean={r.kfold_r2_mean} std={r.kfold_r2_std} />
                <td className="px-3 py-2 text-right text-xs text-muted-foreground">
                  {r.n_train ?? "?"} / {r.n_test ?? "?"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}

function HeaderCell({
  label,
  onClick,
  active,
  dir,
  numeric,
  title,
}: {
  label: string;
  onClick: () => void;
  active: boolean;
  dir: "asc" | "desc";
  numeric?: boolean;
  title?: string;
}) {
  return (
    <th
      className={cn(
        "cursor-pointer select-none px-3 py-2 font-medium hover:bg-accent/40",
        numeric ? "text-right" : "text-left",
      )}
      onClick={onClick}
      title={title}
    >
      <span className="inline-flex items-center gap-1">
        {label}
        {active ? (
          dir === "desc" ? (
            <ArrowDown className="h-3 w-3" />
          ) : (
            <ArrowUp className="h-3 w-3" />
          )
        ) : (
          <ArrowUpDown className="h-3 w-3 opacity-30" />
        )}
      </span>
    </th>
  );
}

function NumCell({
  v,
  digits,
  suffix,
  emphasis,
}: {
  v: number | null | undefined;
  digits: number;
  suffix?: string;
  emphasis?: boolean;
}) {
  if (v == null) {
    return <td className="px-3 py-2 text-right text-muted-foreground">—</td>;
  }
  return (
    <td
      className={cn(
        "px-3 py-2 text-right tabular-nums",
        emphasis && "font-semibold text-emerald-600",
      )}
    >
      {v.toFixed(digits)}
      {suffix}
    </td>
  );
}

function KfoldCell({
  mean,
  std,
}: {
  mean: number | null | undefined;
  std: number | null | undefined;
}) {
  if (mean == null) {
    return <td className="px-3 py-2 text-right text-muted-foreground">—</td>;
  }
  return (
    <td className="px-3 py-2 text-right tabular-nums">
      <span>{mean.toFixed(3)}</span>
      {std != null && (
        <span className="ml-1 text-xs text-muted-foreground">
          ± {std.toFixed(3)}
        </span>
      )}
    </td>
  );
}

function FilterSelect({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <label className="inline-flex items-center gap-1">
      <span className="text-muted-foreground">{label}:</span>
      <select
        className="rounded border bg-background px-1 py-0.5"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        <option value="">всі</option>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}
