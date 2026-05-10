import { Map } from "lucide-react";
import { AppShell } from "@/components/layout/AppShell";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function FieldsPage() {
  return (
    <AppShell>
      <div className="space-y-6">
        <header>
          <h1 className="text-3xl font-bold tracking-tight">Поля</h1>
          <p className="text-muted-foreground">
            Управління сільгоспугіддями — буде реалізовано на тижні 2
          </p>
        </header>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Map className="h-5 w-5 text-primary" />
              Скоро тут буде Leaflet
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground">
            Тиждень 2 додасть інтерактивну карту з можливістю малювати полігони полів
            та зберігати їх у БД (PostgreSQL + PostGIS).
          </CardContent>
        </Card>
      </div>
    </AppShell>
  );
}
