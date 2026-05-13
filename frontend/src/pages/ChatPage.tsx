import { format } from "date-fns";
import { uk } from "date-fns/locale";
import { Loader2, MessageSquare, Trash2 } from "lucide-react";

import { AppShell } from "@/components/layout/AppShell";
import { FaqTab } from "@/components/chat/FaqTab";
import { TemplatesTab } from "@/components/chat/TemplatesTab";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useChatSessions, useDeleteChatSession } from "@/hooks/useChat";
import { useChatStore } from "@/stores/chatStore";

export function ChatPage() {
  return (
    <AppShell>
      <div className="space-y-6">
        <header>
          <h1 className="text-3xl font-bold tracking-tight">AI-помічник</h1>
          <p className="text-muted-foreground">
            База знань про супутникові індекси, ML-прогноз, аномалії та
            готові запитання для чат-бота.
          </p>
        </header>

        <Tabs defaultValue="faq">
          <TabsList>
            <TabsTrigger value="faq">FAQ</TabsTrigger>
            <TabsTrigger value="templates">Шаблони</TabsTrigger>
            <TabsTrigger value="history">Розмови</TabsTrigger>
          </TabsList>

          <TabsContent value="faq">
            <FaqTab />
          </TabsContent>
          <TabsContent value="templates">
            <TemplatesTab />
          </TabsContent>
          <TabsContent value="history">
            <SessionsHistory />
          </TabsContent>
        </Tabs>
      </div>
    </AppShell>
  );
}

function SessionsHistory() {
  const { data, isLoading } = useChatSessions();
  const deleteSession = useDeleteChatSession();
  const { open, setActiveSession } = useChatStore();

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }
  if (!data || data.length === 0) {
    return (
      <Card className="py-12 text-center text-sm text-muted-foreground">
        <MessageSquare className="mx-auto mb-2 h-6 w-6 opacity-50" />
        Розмов ще немає. Перейдіть на вкладку «Шаблони» або відкрийте
        чат-кнопку справа-внизу.
      </Card>
    );
  }

  const handleOpen = (id: number) => {
    setActiveSession(id);
    open();
  };

  return (
    <div className="space-y-2">
      {data.map((s) => (
        <Card key={s.id} className="flex items-center gap-3 p-3">
          <button
            type="button"
            onClick={() => handleOpen(s.id)}
            className="min-w-0 flex-1 text-left"
          >
            <div className="truncate text-sm font-medium">{s.title}</div>
            <div className="text-xs text-muted-foreground">
              Оновлено: {format(new Date(s.updated_at), "d MMM yyyy, HH:mm", { locale: uk })}
              {s.field_id != null && (
                <span className="ml-1">· прив'язано до поля</span>
              )}
            </div>
          </button>
          <Button
            size="icon"
            variant="ghost"
            onClick={() => deleteSession.mutate(s.id)}
            aria-label="Видалити розмову"
          >
            <Trash2 className="h-4 w-4 text-muted-foreground" />
          </Button>
        </Card>
      ))}
    </div>
  );
}
