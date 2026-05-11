import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useAuthStore } from "@/stores/auth";

export interface ChatToolCallEvent {
  name: string;
  args: Record<string, unknown>;
}

export interface ChatStreamState {
  /** Tokens streamed since the user's last message. Cleared on `done`. */
  partial: string;
  /** Tool calls observed in this turn — shown in the UI as collapsible chips. */
  toolCalls: ChatToolCallEvent[];
  /** True while a turn is in flight (between the user's POST and `done`). */
  isStreaming: boolean;
  /** Last failure (if any) from the backend, surfaced inline. */
  error: string | null;
}

const INITIAL: ChatStreamState = {
  partial: "",
  toolCalls: [],
  isStreaming: false,
  error: null,
};

/**
 * Subscribe to the `chat:{session_id}` SSE channel via the backend.
 *
 * We can't use the browser's `EventSource` because it doesn't let us set the
 * `Authorization` header. So we open a fetch + readable-stream and parse the
 * SSE frames ourselves, matching the pattern in `useAlertsStream`.
 *
 * Each frame is `event: <type>\ndata: <json>\n\n`. The shapes we forward to
 * the UI:
 *   - `delta`     — chunk.content appended to `partial`
 *   - `tool_call` — push to `toolCalls`
 *   - `done`      — clear stream state and invalidate messages cache so the
 *     freshly-persisted assistant message is loaded from the DB
 *   - `failed`    — set `error` and stop streaming
 */
export function useChatStream(sessionId: number | null) {
  const [state, setState] = useState<ChatStreamState>(INITIAL);
  const token = useAuthStore((s) => s.accessToken);
  const qc = useQueryClient();

  useEffect(() => {
    if (sessionId == null || !token) {
      setState(INITIAL);
      return;
    }
    const base = import.meta.env.VITE_API_BASE_URL || "";
    const controller = new AbortController();

    void (async () => {
      try {
        const resp = await fetch(`${base}/api/chat/sessions/${sessionId}/stream`, {
          headers: { Authorization: `Bearer ${token}` },
          signal: controller.signal,
        });
        if (!resp.ok || !resp.body) return;
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          while (true) {
            const idx = buf.indexOf("\n\n");
            if (idx === -1) break;
            const frame = buf.slice(0, idx);
            buf = buf.slice(idx + 2);
            handleFrame(frame, setState, () => {
              qc.invalidateQueries({
                queryKey: ["chat", "messages", sessionId],
              });
              qc.invalidateQueries({ queryKey: ["chat", "sessions"] });
            });
          }
        }
      } catch {
        // AbortController.cancel() or network blip — silent retry on remount.
      }
    })();

    return () => controller.abort();
  }, [sessionId, token, qc]);

  return state;
}

function handleFrame(
  frame: string,
  setState: React.Dispatch<React.SetStateAction<ChatStreamState>>,
  onDone: () => void,
) {
  const lines = frame.split("\n");
  let event = "message";
  let dataLine = "";
  for (const line of lines) {
    if (line.startsWith("event: ")) event = line.slice(7);
    else if (line.startsWith("data: ")) dataLine = line.slice(6);
  }
  if (!dataLine) return;
  let payload: Record<string, unknown>;
  try {
    payload = JSON.parse(dataLine);
  } catch {
    return;
  }

  switch (event) {
    case "start":
      setState({ partial: "", toolCalls: [], isStreaming: true, error: null });
      break;
    case "delta": {
      const chunk = String(payload.content ?? "");
      setState((s) => ({ ...s, partial: s.partial + chunk, isStreaming: true }));
      break;
    }
    case "tool_call": {
      const name = String(payload.name ?? "?");
      const args = (payload.args ?? {}) as Record<string, unknown>;
      setState((s) => ({ ...s, toolCalls: [...s.toolCalls, { name, args }] }));
      break;
    }
    case "tool_result":
      // No UI for results — the assistant's content already cites them.
      break;
    case "done":
      setState({ ...INITIAL });
      onDone();
      break;
    case "failed":
      setState({
        partial: "",
        toolCalls: [],
        isStreaming: false,
        error: String(payload.detail ?? payload.error ?? "Помилка"),
      });
      onDone();
      break;
    default:
      break;
  }
}
