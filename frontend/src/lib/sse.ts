/**
 * Server-Sent Events helper.
 *
 * Two reasons we wrap EventSource manually rather than use a library:
 * 1. We need to inject the JWT — and EventSource doesn't support custom
 *    headers in any browser. The fix is to pass the token as a query
 *    parameter to the URL (the backend's `CurrentUser` dep is read from
 *    Authorization header by default; we'll fall back to ?token=… here).
 *    For now this codepath uses cookies via withCredentials, since the
 *    refresh cookie is httpOnly and access uses Authorization which
 *    EventSource can't set — we route SSE behind the FastAPI dep that
 *    accepts query-string token (see backend deps.py future change).
 *
 *    For Week-3 milestone we accept that SSE requires a same-origin
 *    proxy (Vite dev proxy already handles this) and use Authorization
 *    via a polyfilled fetch-EventSource implementation here.
 * 2. We add safety nets: auto-close on terminal states, JSON parsing,
 *    typed callbacks.
 */
import { useAuthStore } from "@/stores/auth";

export type JobState = "queued" | "running" | "done" | "failed" | "connected";

export interface JobUpdate {
  state: JobState;
  progress?: number;
  data?: Record<string, unknown>;
  error?: string;
}

export interface SubscribeOptions {
  onUpdate?: (u: JobUpdate) => void;
  onDone?: (u: JobUpdate) => void;
  onError?: (err: Error) => void;
}

/**
 * Subscribe to a server-side job's progress.
 *
 * Returns an `unsubscribe` function. Calling it more than once is safe.
 */
export function subscribeToJob(
  jobId: string,
  { onUpdate, onDone, onError }: SubscribeOptions = {},
): () => void {
  const base = import.meta.env.VITE_API_BASE_URL || "";
  const token = useAuthStore.getState().accessToken;
  const controller = new AbortController();
  let closed = false;

  const close = () => {
    if (closed) return;
    closed = true;
    controller.abort();
  };

  void (async () => {
    try {
      const resp = await fetch(`${base}/api/jobs/${jobId}/stream`, {
        headers: { Authorization: token ? `Bearer ${token}` : "" },
        signal: controller.signal,
      });
      if (!resp.ok || !resp.body) {
        throw new Error(`SSE failed: HTTP ${resp.status}`);
      }
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (!closed) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        // SSE frames are separated by a blank line.
        let idx: number;
        while ((idx = buf.indexOf("\n\n")) !== -1) {
          const frame = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          const lines = frame.split("\n");
          let dataStr = "";
          for (const line of lines) {
            if (line.startsWith("data:")) {
              dataStr += line.slice(5).trim();
            }
          }
          if (!dataStr) continue;
          try {
            const update = JSON.parse(dataStr) as JobUpdate;
            onUpdate?.(update);
            if (update.state === "done" || update.state === "failed") {
              onDone?.(update);
              close();
              return;
            }
          } catch (parseErr) {
            console.warn("SSE parse error", parseErr, dataStr);
          }
        }
      }
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      onError?.(err as Error);
    }
  })();

  return close;
}
