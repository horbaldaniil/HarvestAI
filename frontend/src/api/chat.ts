import { api } from "./client";

export type ChatRole = "user" | "assistant" | "tool" | "system";

export interface ChatSessionRead {
  id: number;
  user_id: number;
  field_id: number | null;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ChatToolCall {
  id: string | null;
  name: string;
  args: string;
}

export interface ChatMessageRead {
  id: number;
  session_id: number;
  role: ChatRole;
  content: string;
  tool_calls: ChatToolCall[] | null;
  tool_call_id: string | null;
  tool_name: string | null;
  tokens_in: number | null;
  tokens_out: number | null;
  created_at: string;
}

export async function listSessions(): Promise<ChatSessionRead[]> {
  const { data } = await api.get<ChatSessionRead[]>("/api/chat/sessions");
  return data;
}

export async function createSession(
  fieldId: number | null = null,
  title?: string,
): Promise<ChatSessionRead> {
  const { data } = await api.post<ChatSessionRead>("/api/chat/sessions", {
    field_id: fieldId,
    title,
  });
  return data;
}

export async function listMessages(sessionId: number): Promise<ChatMessageRead[]> {
  const { data } = await api.get<ChatMessageRead[]>(
    `/api/chat/sessions/${sessionId}/messages`,
  );
  return data;
}

export async function sendMessage(
  sessionId: number,
  content: string,
): Promise<ChatMessageRead> {
  const { data } = await api.post<ChatMessageRead>(
    `/api/chat/sessions/${sessionId}/messages`,
    { content },
  );
  return data;
}

export async function deleteSession(sessionId: number): Promise<void> {
  await api.delete(`/api/chat/sessions/${sessionId}`);
}
