import { useEffect } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import * as api from "@/api/alerts";
import { useAuthStore } from "@/stores/auth";

const KEY = ["alerts"] as const;
const COUNT_KEY = ["alerts", "count"] as const;

export function useAlerts(acknowledged = false) {
  return useQuery({
    queryKey: [...KEY, acknowledged] as const,
    queryFn: () => api.listAlerts(acknowledged),
    staleTime: 30 * 1000,
    refetchInterval: 60 * 1000,
  });
}

export function useAlertsCount() {
  return useQuery({
    queryKey: COUNT_KEY,
    queryFn: api.alertsCount,
    staleTime: 15 * 1000,
    refetchInterval: 30 * 1000,
  });
}

export function useAcknowledgeAlert() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (alertId: number) => api.acknowledgeAlert(alertId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: KEY });
      qc.invalidateQueries({ queryKey: COUNT_KEY });
    },
  });
}

/**
 * SSE-driven cache invalidation for the alerts bell. Subscribes once at
 * mount of any component that calls this hook (e.g. the AlertsBell), so
 * new alerts pushed from the worker propagate to the badge within seconds
 * without explicit polling.
 */
export function useAlertsStream() {
  const qc = useQueryClient();
  const token = useAuthStore((s) => s.accessToken);

  useEffect(() => {
    if (!token) return;
    const base = import.meta.env.VITE_API_BASE_URL || "";
    const controller = new AbortController();

    void (async () => {
      try {
        const resp = await fetch(`${base}/api/alerts/stream`, {
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
            const isAlertFrame = frame.includes("event: alert");
            if (isAlertFrame) {
              qc.invalidateQueries({ queryKey: KEY });
              qc.invalidateQueries({ queryKey: COUNT_KEY });
              toast.info("Нове сповіщення");
            }
          }
        }
      } catch {
        // Ignore — re-tries on remount/auth change.
      }
    })();

    return () => controller.abort();
  }, [token, qc]);
}
