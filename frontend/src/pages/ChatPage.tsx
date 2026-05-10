import { MessageSquare } from "lucide-react";
import { AppShell } from "@/components/layout/AppShell";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function ChatPage() {
  return (
    <AppShell>
      <div className="space-y-6">
        <header>
          <h1 className="text-3xl font-bold tracking-tight">AI-помічник</h1>
          <p className="text-muted-foreground">
            Чат-бот з GPT-4o-mini — буде реалізовано на тижні 5
          </p>
        </header>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <MessageSquare className="h-5 w-5 text-primary" />
              Розумний агроном
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground">
            AI-помічник зможе відповідати на питання типу "Чому моя пшениця жовтіє?",
            використовуючи реальні дані з ваших полів (NDVI, погода, аномалії).
          </CardContent>
        </Card>
      </div>
    </AppShell>
  );
}
