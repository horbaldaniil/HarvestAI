import { AppShell } from "@/components/layout/AppShell";
import { ReportBuilder } from "@/components/reports/ReportBuilder";
import { ReportsHistory } from "@/components/reports/ReportsHistory";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export function ReportsPage() {
  return (
    <AppShell>
      <div className="space-y-6">
        <header>
          <h1 className="text-3xl font-bold tracking-tight">Звіти</h1>
          <p className="text-muted-foreground">
            Згенеруйте PDF звіт із кастомізованими секціями, або перегляньте
            раніше створені.
          </p>
        </header>

        <Tabs defaultValue="builder">
          <TabsList>
            <TabsTrigger value="builder">Builder</TabsTrigger>
            <TabsTrigger value="history">Історія</TabsTrigger>
          </TabsList>
          <TabsContent value="builder">
            <ReportBuilder />
          </TabsContent>
          <TabsContent value="history">
            <ReportsHistory />
          </TabsContent>
        </Tabs>
      </div>
    </AppShell>
  );
}
