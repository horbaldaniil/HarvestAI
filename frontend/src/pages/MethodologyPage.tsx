import { Loader2, AlertTriangle } from "lucide-react";

import { AppShell } from "@/components/layout/AppShell";
import { AlgorithmComparisonTable } from "@/components/methodology/AlgorithmComparisonTable";
import { MetricsCharts } from "@/components/methodology/MetricsCharts";
import { OblastCoverageMap } from "@/components/methodology/OblastCoverageMap";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useMethodologyOverview } from "@/hooks/useMethodology";

export function MethodologyPage() {
  const { data, isLoading, error } = useMethodologyOverview();

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
        <header>
          <h1 className="text-3xl font-bold tracking-tight">Методологія</h1>
          <p className="text-muted-foreground">
            Джерела даних, алгоритми, метрики — повна прозорість ML-стеку
            HarvestAI для академічного аудиту.
          </p>
        </header>

        <Tabs defaultValue="data">
          <TabsList>
            <TabsTrigger value="data">Дані</TabsTrigger>
            <TabsTrigger value="algorithms">Алгоритми</TabsTrigger>
            <TabsTrigger value="metrics">Метрики</TabsTrigger>
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

          <TabsContent value="algorithms">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Порівняння алгоритмів</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <p className="text-muted-foreground">
                  Тренуємо три родини моделей на однаковому наборі ознак
                  (17 фіч: 11 вегетаційних + 4 погодних + 2 геопросторових).
                  Це чесний methods-section: можна оцінити чи додаткова
                  складність LSTM/трансформера виправдана у нашому контексті.
                </p>
                <AlgorithmComparisonTable models={data.models} />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="metrics">
            <MetricsCharts models={data.models} />
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
