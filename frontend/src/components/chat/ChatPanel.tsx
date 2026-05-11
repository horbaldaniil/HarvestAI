import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { History, Plus, Send, Trash2, X } from "lucide-react";

import { sendMessage as postChatMessage } from "@/api/chat";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { MessageBubble, StreamingBubble } from "./MessageBubble";
import { SuggestedPrompts } from "./SuggestedPrompts";
import { SessionsList } from "./SessionsList";
import {
  useChatMessages,
  useCreateChatSession,
  useDeleteChatSession,
  useSendChatMessage,
} from "@/hooks/useChat";
import { useChatStream } from "@/hooks/useChatStream";
import { useChatStore } from "@/stores/chatStore";

export function ChatPanel() {
  const { t } = useTranslation();
  const {
    isOpen,
    close,
    activeSessionId,
    setActiveSession,
    contextFieldId,
    pendingPrompt,
    clearPendingPrompt,
  } = useChatStore();
  const [showHistory, setShowHistory] = useState(false);

  const createSession = useCreateChatSession();
  const deleteSession = useDeleteChatSession();
  const messages = useChatMessages(activeSessionId);
  const sendMessage = useSendChatMessage(activeSessionId);
  const stream = useChatStream(isOpen ? activeSessionId : null);

  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  // Auto-scroll to the latest message on data change or partial update.
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.data, stream.partial, stream.toolCalls.length]);

  // Auto-send pendingPrompt from chatStore (e.g. Templates tab in /chat).
  // We create a session first if none is active, then fire the message via
  // the same path the user would when typing manually. The store field is
  // cleared regardless of success so a failure doesn't loop on next render.
  useEffect(() => {
    if (!pendingPrompt || !isOpen) return;
    const prompt = pendingPrompt;
    clearPendingPrompt();
    void (async () => {
      try {
        let sid = activeSessionId;
        if (sid == null) {
          const sess = await createSession.mutateAsync({ fieldId: contextFieldId });
          sid = sess.id;
          setActiveSession(sid);
        }
        // sendMessage (mutation) is bound to activeSessionId via closure
        // and would fire BEFORE setActiveSession finishes propagating, so
        // we call the raw API with the just-created sid directly.
        await postChatMessage(sid, prompt);
      } catch {
        // Errors surface via the SSE stream or the existing toast in useSendChatMessage.
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingPrompt, isOpen]);

  const handleNewSession = async () => {
    const sess = await createSession.mutateAsync({ fieldId: contextFieldId });
    setActiveSession(sess.id);
    setShowHistory(false);
  };

  const handleSend = async () => {
    const text = input.trim();
    if (!text || stream.isStreaming) return;
    let sid = activeSessionId;
    if (sid == null) {
      const sess = await createSession.mutateAsync({ fieldId: contextFieldId });
      sid = sess.id;
      setActiveSession(sid);
    }
    setInput("");
    await sendMessage.mutateAsync(text);
  };

  const handleSuggested = async (prompt: string) => {
    setInput(prompt);
    setTimeout(() => void handleSend(), 0);
  };

  const handleDelete = async (id: number) => {
    await deleteSession.mutateAsync(id);
    if (activeSessionId === id) setActiveSession(null);
  };

  if (!isOpen) return null;

  const list = messages.data ?? [];
  const showSuggested = !showHistory && list.length === 0 && !stream.isStreaming;

  return (
    <div
      className={cn(
        // Below FloatingChatButton's z-index when closed; above when open.
        "fixed bottom-6 right-6 z-[1950] flex h-[600px] w-[420px] max-h-[calc(100vh-3rem)] max-w-[calc(100vw-3rem)]",
        "flex-col overflow-hidden rounded-xl border bg-card text-card-foreground shadow-2xl",
      )}
    >
      <header className="flex items-center justify-between border-b bg-primary px-4 py-2 text-primary-foreground">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold">{t("chat.title")}</h2>
          {activeSessionId == null && (
            <span className="text-xs opacity-80">· {t("chat.newSession")}</span>
          )}
        </div>
        <div className="flex items-center gap-1">
          <Button
            size="icon"
            variant="ghost"
            className="h-7 w-7 text-primary-foreground hover:bg-primary-foreground/10"
            onClick={() => setShowHistory((v) => !v)}
            aria-label={t("chat.history")}
          >
            <History className="h-4 w-4" />
          </Button>
          <Button
            size="icon"
            variant="ghost"
            className="h-7 w-7 text-primary-foreground hover:bg-primary-foreground/10"
            onClick={handleNewSession}
            aria-label={t("chat.newSession")}
            disabled={createSession.isPending}
          >
            <Plus className="h-4 w-4" />
          </Button>
          {activeSessionId != null && (
            <Button
              size="icon"
              variant="ghost"
              className="h-7 w-7 text-primary-foreground hover:bg-primary-foreground/10"
              onClick={() => handleDelete(activeSessionId)}
              aria-label={t("chat.deleteSession")}
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          )}
          <Button
            size="icon"
            variant="ghost"
            className="h-7 w-7 text-primary-foreground hover:bg-primary-foreground/10"
            onClick={close}
            aria-label={t("common.close")}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </header>

      <div className="flex flex-1 flex-col gap-2 overflow-y-auto bg-background p-3">
        {showHistory ? (
          <SessionsList
            onSelect={(id) => {
              setActiveSession(id);
              setShowHistory(false);
            }}
            onDelete={handleDelete}
          />
        ) : (
          <>
            {list.length === 0 && !stream.isStreaming && (
              <div className="rounded-md bg-muted/50 p-3 text-xs text-muted-foreground">
                {t("chat.greeting")}
              </div>
            )}
            {list.map((m) => (
              <MessageBubble key={m.id} message={m} />
            ))}
            {stream.isStreaming && (
              <StreamingBubble
                content={stream.partial}
                toolNames={stream.toolCalls.map((tc) => tc.name)}
              />
            )}
            {stream.error && (
              <div className="rounded-md border border-destructive/50 bg-destructive/10 p-2 text-xs text-destructive">
                {t("chat.errorGeneric")}: {stream.error}
              </div>
            )}
            {showSuggested && (
              <div className="mt-2">
                <div className="mb-2 text-xs text-muted-foreground">
                  {t("chat.suggestedLabel")}
                </div>
                <SuggestedPrompts onPick={handleSuggested} disabled={sendMessage.isPending} />
              </div>
            )}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      <footer className="border-t bg-card p-2">
        <form
          className="flex items-end gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            void handleSend();
          }}
        >
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void handleSend();
              }
            }}
            placeholder={t("chat.placeholder")}
            rows={1}
            className="flex max-h-32 min-h-9 flex-1 resize-none rounded-md border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
            disabled={stream.isStreaming || sendMessage.isPending}
          />
          <Button
            type="submit"
            size="icon"
            className="h-9 w-9 shrink-0"
            disabled={!input.trim() || stream.isStreaming || sendMessage.isPending}
            aria-label={t("chat.send")}
          >
            <Send className="h-4 w-4" />
          </Button>
        </form>
      </footer>
    </div>
  );
}
