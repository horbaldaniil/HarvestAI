import { type ReactNode } from "react";
import { AppShell } from "@/components/layout/AppShell";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface PlaceholderPageProps {
  title: string;
  description: string;
  icon: ReactNode;
}

export function PlaceholderPage({ title, description, icon }: PlaceholderPageProps) {
  return (
    <AppShell>
      <div className="space-y-6">
        <header>
          <h1 className="text-3xl font-bold tracking-tight">{title}</h1>
        </header>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">{icon} Скоро</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground">{description}</CardContent>
        </Card>
      </div>
    </AppShell>
  );
}
