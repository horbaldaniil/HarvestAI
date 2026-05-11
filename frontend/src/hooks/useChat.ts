import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import * as api from "@/api/chat";

const SESSIONS_KEY = ["chat", "sessions"] as const;
const messagesKey = (id: number | null) => ["chat", "messages", id] as const;

export function useChatSessions(enabled: boolean = true) {
  return useQuery({
    queryKey: SESSIONS_KEY,
    queryFn: api.listSessions,
    enabled,
    staleTime: 30 * 1000,
  });
}

export function useChatMessages(sessionId: number | null) {
  return useQuery({
    queryKey: messagesKey(sessionId),
    queryFn: () => api.listMessages(sessionId!),
    enabled: sessionId != null,
    staleTime: 0,
  });
}

export function useCreateChatSession() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (vars: { fieldId?: number | null; title?: string }) =>
      api.createSession(vars.fieldId ?? null, vars.title),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: SESSIONS_KEY });
    },
  });
}

export function useSendChatMessage(sessionId: number | null) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (content: string) => {
      if (sessionId == null) throw new Error("no session");
      return api.sendMessage(sessionId, content);
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: messagesKey(sessionId) });
      qc.invalidateQueries({ queryKey: SESSIONS_KEY });
    },
    onError: (err: unknown) => {
      const e = err as { response?: { status?: number; data?: { detail?: string } } };
      if (e.response?.status === 429) {
        toast.error(e.response.data?.detail ?? "Перевищено ліміт запитів.");
      } else {
        toast.error("Не вдалося надіслати повідомлення.");
      }
    },
  });
}

export function useDeleteChatSession() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (sessionId: number) => api.deleteSession(sessionId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: SESSIONS_KEY });
    },
  });
}
