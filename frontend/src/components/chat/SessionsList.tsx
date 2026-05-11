import { format } from "date-fns";
import { uk } from "date-fns/locale";
import { Loader2, Trash2 } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import { useChatSessions } from "@/hooks/useChat";

interface SessionsListProps {
  onSelect: (sessionId: number) => void;
  onDelete: (sessionId: number) => void;
}

export function SessionsList({ onSelect, onDelete }: SessionsListProps) {
  const { t } = useTranslation();
  const { data, isLoading } = useChatSessions();

  if (isLoading) {
    return (
      <div className="flex justify-center py-8">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <div className="py-8 text-center text-sm text-muted-foreground">
        {t("chat.noSessions")}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1">
      {data.map((s) => (
        <div
          key={s.id}
          className="group flex items-center gap-2 rounded-md border bg-card p-2 hover:bg-accent/30"
        >
          <button
            type="button"
            onClick={() => onSelect(s.id)}
            className="flex-1 text-left"
          >
            <div className="truncate text-sm font-medium">{s.title}</div>
            <div className="text-[10px] text-muted-foreground">
              {format(new Date(s.updated_at), "d MMM, HH:mm", { locale: uk })}
              {s.field_id != null && (
                <span className="ml-1">· {t("chat.fieldScoped")}</span>
              )}
            </div>
          </button>
          <Button
            size="icon"
            variant="ghost"
            className="h-7 w-7 opacity-0 transition-opacity group-hover:opacity-100"
            onClick={(e) => {
              e.stopPropagation();
              onDelete(s.id);
            }}
            aria-label={t("chat.deleteSession")}
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </div>
      ))}
    </div>
  );
}
