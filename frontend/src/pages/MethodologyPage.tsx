import { useState } from "react";
import { Loader2, AlertTriangle, Download } from "lucide-react";
import { toast } from "sonner";

import { downloadMethodologyReport } from "@/api/methodology";
import { AppShell } from "@/components/layout/AppShell";
import { AlgorithmComparisonTable } from "@/components/methodology/AlgorithmComparisonTable";
import { AlgorithmLeaderboard } from "@/components/methodology/AlgorithmLeaderboard";
import { CalibrationPlot } from "@/components/methodology/CalibrationPlot";
import { GlobalShapSummary } from "@/components/methodology/GlobalShapSummary";
import { LearningCurves } from "@/components/methodology/LearningCurves";
import { MetricsCharts } from "@/components/methodology/MetricsCharts";
import { OblastCoverageMap } from "@/components/methodology/OblastCoverageMap";
import { OblastResidualMap } from "@/components/methodology/OblastResidualMap";
import { PartialDependenceCharts } from "@/components/methodology/PartialDependenceCharts";
import { PerCropResidualScatter } from "@/components/methodology/PerCropResidualScatter";
import { V4AblationChart } from "@/components/methodology/V4AblationChart";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useMethodologyOverview } from "@/hooks/useMethodology";

export function MethodologyPage() {
  const { data, isLoading, error } = useMethodologyOverview();
  const [downloadingPdf, setDownloadingPdf] = useState(false);

  const handleDownloadPdf = async () => {
    setDownloadingPdf(true);
    try {
      // Default (wheat, stack) — the page-level button doesn't expose
      // selectors; users wanting other combos can curl the endpoint
      // directly. Sensible default for thesis-defense moment.
      await downloadMethodologyReport("wheat", "stack");
      toast.success("Методологію PDF завантажено");
    } catch (err) {
      console.error(err);
      toast.error("Не вдалося завантажити PDF");
    } finally {
      setDownloadingPdf(false);
    }
  };

  if (isLoading) {
    return (
      <AppShell>
        <div className="flex h-64 items-center justify-center">
          <Loader2 className="h-6 w-6 animate-spin text-primary" />
        </div>
      </AppShell>
    );
  }
  if (error || !data) {
    return (
      <AppShell>
        <Card>
          <CardContent className="flex items-center gap-3 py-10 text-sm text-muted-foreground">
            <AlertTriangle className="h-5 w-5" />
            Не вдалося завантажити методологію.
          </CardContent>
        </Card>
      </AppShell>
    );
  }

  return (
    <AppShell>
      <div className="space-y-6">
        <header className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Методологія</h1>
            <p className="text-muted-foreground">
              Джерела даних, алгоритми, метрики — повна прозорість ML-стеку
              HarvestAI для академічного аудиту.
            </p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={handleDownloadPdf}
            disabled={downloadingPdf}
            className="shrink-0"
            title="Завантажити повну методологію як PDF (wheat × stack за замовчуванням)"
          >
            {downloadingPdf ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Download className="h-4 w-4" />
            )}
            <span className="ml-2">PDF</span>
          </Button>
        </header>

        <Tabs defaultValue="data">
          <TabsList>
            <TabsTrigger value="data">Дані</TabsTrigger>
            <TabsTrigger value="algorithms">Алгоритми</TabsTrigger>
            <TabsTrigger value="metrics">Метрики</TabsTrigger>
            <TabsTrigger value="leaderboard">Лідерборд</TabsTrigger>
            <TabsTrigger value="residuals">Карта помилок</TabsTrigger>
            <TabsTrigger value="explainability">Інтерпретованість</TabsTrigger>
            <TabsTrigger value="learning">Криві навчання</TabsTrigger>
            <TabsTrigger value="discussion">Чесна оцінка</TabsTrigger>
          </TabsList>

          <TabsContent value="data" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Джерела даних</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="list-disc space-y-1 pl-5 text-sm">
                  {data.data_sources.map((s) => (
                    <li key={s}>{s}</li>
                  ))}
                </ul>
                <div className="mt-4 grid gap-2 text-sm sm:grid-cols-2">
                  <Fact label="Областей" value={data.oblast_count.toString()} />
                  <Fact
                    label="Зразків на область"
                    value={
                      data.sample_count_per_oblast?.toString() ?? "—"
                    }
                  />
                  <Fact label="Період" value={data.year_range.join(" – ")} />
                  <Fact label="Розбиття" value={data.train_val_test_split} />
                  <Fact label="Активна модель" value={data.active_algorithm} />
                </div>
              </CardContent>
            </Card>

            <OblastCoverageMap />
          </TabsContent>

          <TabsContent value="algorithms" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Порівняння алгоритмів</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <p className="text-muted-foreground">
                  П'ять родин моделей на однаковому наборі ознак
                  (17 фіч: 11 вегетаційних + 4 погодних + 2 геопросторових):
                  Random Forest, XGBoost, LightGBM, LSTM та Stacked
                  Ensemble (Ridge meta-learner). Per-crop drill-down
                  з sparkline RepeatedKFold(5×3) варіації — компактна
                  ілюстрація bias/variance trade-off між моделями.
                </p>
                <AlgorithmComparisonTable models={data.models} />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="metrics" className="space-y-4">
            <MetricsCharts models={data.models} />
            <CalibrationPlot />
          </TabsContent>

          <TabsContent value="leaderboard" className="space-y-4">
            <p className="text-sm text-muted-foreground">
              Усі натреновані моделі v4 (5 family × 13 культур) в одній
              сортовній таблиці. Стандартний DS-pattern: один рядок на
              (культура × модель), стовпці —{" "}
              <strong>test R², RMSE, MAE, MAPE, LOOCV R²</strong> та
              RepeatedKFold(5×3) варіація. Натисніть на заголовок щоб
              відсортувати; використовуйте фільтри щоб звузити
              перегляд.
            </p>
            <V4AblationChart />
            <AlgorithmLeaderboard />
          </TabsContent>

          <TabsContent value="residuals" className="space-y-4">
            <p className="text-sm text-muted-foreground">
              Mean absolute residual на 2023 test-split, агреговано по
              областях. Карта показує <em>де</em> модель помиляється
              найбільше; scatter показує <em>як саме</em> вона помиляється
              (over-/under-prediction). Разом — найсильніший візуальний
              аргумент про spatial generalisation
              (Roberts et al. 2017).
            </p>
            <OblastResidualMap />
            <PerCropResidualScatter />
          </TabsContent>

          <TabsContent value="explainability" className="space-y-4">
            <p className="text-sm text-muted-foreground">
              Три незалежні підходи до інтерпретованості:{" "}
              <strong>SHAP TreeExplainer</strong> (Lundberg &amp; Lee 2017),{" "}
              <strong>Permutation Importance</strong> (Breiman 2001) та{" "}
              <strong>1-D Partial Dependence Plots</strong> (Friedman 2001).
              Якщо всі три виділяють однакові топ-фічі — модель справді
              на них опирається, а не "вгадує" статистичні корелянти.
              PDP додатково показує <em>напрямок</em> впливу (монотонний vs
              U-shape vs пороговий).
            </p>
            <GlobalShapSummary />
            <PartialDependenceCharts />
          </TabsContent>

          <TabsContent value="learning" className="space-y-3">
            <p className="text-sm text-muted-foreground">
              Test R² як функція обсягу тренувальних даних. Якщо крива
              ще піднімається на 100 % — додавання даних допоможе;
              якщо плато — bias-limited (треба інші фічі, не більше
              рядків).
            </p>
            <LearningCurves />
          </TabsContent>

          <TabsContent value="discussion">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Що працює і що ні</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm leading-relaxed">
                <p>
                  <strong>Сильні сторони.</strong> Інтерпретовані прогнози
                  через SHAP (top-5 факторів), чесні довірчі інтервали через
                  quantile regression, цілковита прозорість методології
                  (включно з цією сторінкою).
                </p>
                <p>
                  <strong>Обмеження.</strong> Yield labels — на рівні
                  oblast×year (USDA FAS), не на рівні поля. Це означає що R²
                  на тесті обмежений тим, наскільки oblast-середня
                  врожайність репрезентує конкретні поля. Перехід на
                  field-level USDA NASS або українську DSS-агрі бази —
                  природне розширення для magistr-роботи.
                </p>
                <p>
                  <strong>Дисциплінованість.</strong> Розбиття за роками
                  (2019-21 / 22 / 23) запобігає leakage що буває коли train
                  і test містять той самий oblast в різні роки. R² на 2023
                  — фактичний out-of-time generalization. Усі моделі
                  оцінювалися за однаковою процедурою.
                </p>
                <p>
                  <strong>Подальші напрямки.</strong> See{" "}
                  <code className="rounded bg-muted px-1 text-xs">
                    THESIS_EXTENSIONS.md
                  </code>{" "}
                  for the magistr-level extension hooks (DL families,
                  hyperspectral fusion, IoT sensors, federated learning,
                  multi-tenant SaaS).
                </p>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </AppShell>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-2 rounded-md border bg-card p-2">
      <span className="text-xs text-muted-foreground">{label}:</span>
      <span className="font-medium">{value}</span>
    </div>
  );
}
