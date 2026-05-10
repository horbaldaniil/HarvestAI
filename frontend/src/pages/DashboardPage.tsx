import { useTranslation } from "react-i18next";
import { Map, TrendingUp, AlertTriangle, Leaf } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AppShell } from "@/components/layout/AppShell";

export function DashboardPage() {
  const { t } = useTranslation();

  // v0: placeholder metrics. Real values arrive in week 4 (dashboard router).
  const stats = [
    { key: "totalFields", icon: Map, value: "—", color: "text-harvest-600" },
    { key: "totalArea", icon: Leaf, value: "— га", color: "text-harvest-700" },
    { key: "activeAlerts", icon: AlertTriangle, value: "—", color: "text-amber-600" },
    { key: "avgNdvi", icon: TrendingUp, value: "—", color: "text-emerald-600" },
  ] as const;

  return (
    <AppShell>
      <div className="space-y-6">
        <header>
          <h1 className="text-3xl font-bold tracking-tight">{t("dashboard.title")}</h1>
          <p className="text-muted-foreground">{t("app.tagline")}</p>
        </header>

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {stats.map(({ key, icon: Icon, value, color }) => (
            <Card key={key}>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  {t(`dashboard.${key}` as const)}
                </CardTitle>
                <Icon className={`h-5 w-5 ${color}`} />
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">{value}</div>
              </CardContent>
            </Card>
          ))}
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Ласкаво просимо до HarvestAI 🌾</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4 text-sm text-muted-foreground">
            <p>
              Це початковий скаффолд проєкту. Наступні етапи розробки:
            </p>
            <ul className="ml-5 list-disc space-y-1">
              <li>Тиждень 2: Управління полями — малювання полігонів на Leaflet</li>
              <li>Тиждень 3: Інтеграція Sentinel Hub, розрахунок NDVI/EVI/NDWI/SAVI</li>
              <li>Тиждень 4: ML-прогноз врожайності, виявлення аномалій</li>
              <li>Тиждень 5: AI-помічник з OpenAI, PDF-звіти</li>
              <li>Тиждень 6: Полірування UI, тести, документація</li>
            </ul>
          </CardContent>
        </Card>
      </div>
    </AppShell>
  );
}
