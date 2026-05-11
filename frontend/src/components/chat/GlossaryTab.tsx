import { useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Search, X } from "lucide-react";

import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  GLOSSARY,
  GLOSSARY_CATEGORY_LABELS,
  type GlossaryEntry,
  type GlossaryCategory,
} from "@/lib/knowledge";
import { useChatStore } from "@/stores/chatStore";
import { Button } from "@/components/ui/button";

export function GlossaryTab() {
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<GlossaryEntry | null>(null);
  const openWithPrompt = useChatStore((s) => s.openWithPrompt);

  const grouped = useMemo(() => {
    const q = query.trim().toLowerCase();
    const filtered = q
      ? GLOSSARY.filter(
          (e) =>
            e.title.toLowerCase().includes(q) ||
            e.short.toLowerCase().includes(q) ||
            e.body.toLowerCase().includes(q),
        )
      : GLOSSARY;
    const byCat = new Map<GlossaryCategory, GlossaryEntry[]>();
    for (const entry of filtered) {
      const arr = byCat.get(entry.category) ?? [];
      arr.push(entry);
      byCat.set(entry.category, arr);
    }
    return byCat;
  }, [query]);

  const askBot = (entry: GlossaryEntry) => {
    setSelected(null);
    openWithPrompt(
      `Розкажи мені детальніше про "${entry.title}" у контексті моїх полів. Якщо доречно — наведи приклади з моїх даних.`,
    );
  };

  return (
    <div className="space-y-4">
      <div className="relative">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          placeholder="Пошук за назвою, описом..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="pl-9"
        />
        {query && (
          <button
            type="button"
            onClick={() => setQuery("")}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      {grouped.size === 0 && (
        <Card>
          <CardContent className="py-8 text-center text-sm text-muted-foreground">
            Нічого не знайдено за запитом «{query}».
          </CardContent>
        </Card>
      )}

      {Array.from(grouped.entries()).map(([cat, entries]) => (
        <section key={cat}>
          <h3 className="mb-2 text-sm font-semibold text-muted-foreground">
            {GLOSSARY_CATEGORY_LABELS[cat]}
          </h3>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {entries.map((entry) => (
              <Card
                key={entry.id}
                className="cursor-pointer transition-colors hover:bg-accent/30"
                onClick={() => setSelected(entry)}
              >
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">{entry.title}</CardTitle>
                </CardHeader>
                <CardContent className="text-xs text-muted-foreground">
                  {entry.short}
                </CardContent>
              </Card>
            ))}
          </div>
        </section>
      ))}

      <Dialog
        open={selected != null}
        onOpenChange={(open) => !open && setSelected(null)}
      >
        <DialogContent className="max-w-2xl">
          {selected && (
            <>
              <DialogHeader>
                <DialogTitle>{selected.title}</DialogTitle>
                {selected.short && (
                  <p className="text-sm text-muted-foreground">{selected.short}</p>
                )}
              </DialogHeader>
              <div className="markdown-body max-h-[60vh] overflow-y-auto break-words">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {selected.body}
                </ReactMarkdown>
              </div>
              <div className="flex justify-end">
                <Button size="sm" onClick={() => askBot(selected)}>
                  Запитати у бота
                </Button>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
