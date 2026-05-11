import { useMemo } from "react";
import { Sparkles } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TEMPLATES, type TemplateEntry } from "@/lib/knowledge";
import { useChatStore } from "@/stores/chatStore";

const CATEGORY_LABELS: Record<string, string> = {
  overview: "Огляд",
  analysis: "Аналіз",
  planning: "Планування",
  advisory: "Поради",
};

export function TemplatesTab() {
  const openWithPrompt = useChatStore((s) => s.openWithPrompt);

  const grouped = useMemo(() => {
    const byCat = new Map<string, TemplateEntry[]>();
    for (const t of TEMPLATES) {
      const arr = byCat.get(t.category) ?? [];
      arr.push(t);
      byCat.set(t.category, arr);
    }
    return byCat;
  }, []);

  return (
    <div className="space-y-6">
      <p className="text-sm text-muted-foreground">
        Готові запитання до AI-помічника. Клік відкриває чат і одразу
        надсилає запит — бот викличе потрібні інструменти і відповість
        по вашим конкретним полям.
      </p>

      {Array.from(grouped.entries()).map(([cat, entries]) => (
        <section key={cat}>
          <h3 className="mb-2 text-sm font-semibold text-muted-foreground">
            {CATEGORY_LABELS[cat] ?? cat}
          </h3>
          <div className="grid gap-3 sm:grid-cols-2">
            {entries.map((tpl) => (
              <Card key={tpl.id} className="flex flex-col">
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-start gap-2 text-sm">
                    <Sparkles className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
                    <span>{tpl.label}</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="flex flex-1 flex-col justify-between gap-3">
                  <p className="text-xs leading-snug text-muted-foreground">
                    {tpl.prompt}
                  </p>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => openWithPrompt(tpl.prompt)}
                  >
                    Запитати
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}
