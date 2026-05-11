import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { api } from "@/api/client";
import * as obsApi from "@/api/observations";
import type { IndexName } from "@/api/observations";
import { subscribeToJob, type JobUpdate } from "@/lib/sse";

const KEY = (fieldId: number) => ["observations", fieldId] as const;
const HEATMAP_KEY = (fieldId: number, date: string, index: IndexName) =>
  ["heatmap-blob", fieldId, date, index] as const;

export function useObservations(fieldId: number | null | undefined) {
  return useQuery({
    queryKey: fieldId ? KEY(fieldId) : ["observations", "disabled"],
    queryFn: () => obsApi.listObservations(fieldId!),
    enabled: typeof fieldId === "number",
    staleTime: 60 * 1000,
  });
}

/**
 * Trigger refresh + listen to the resulting SSE stream for progress.
 *
 * Exposes both the mutation (to invoke .mutate(fieldId)) and live job state.
 */
export function useRefreshObservations() {
  const qc = useQueryClient();
  const [progress, setProgress] = useState<JobUpdate | null>(null);
  const [activeFieldId, setActiveFieldId] = useState<number | null>(null);

  const mutation = useMutation({
    mutationFn: (fieldId: number) => obsApi.refreshObservations(fieldId),
    onSuccess: (data, fieldId) => {
      setActiveFieldId(fieldId);
      setProgress({ state: "queued" });
      subscribeToJob(data.job_id, {
        onUpdate: setProgress,
        onDone: (update) => {
          if (update.state === "done") {
            const rows = (update.data?.rows as number | undefined) ?? 0;
            qc.invalidateQueries({ queryKey: KEY(fieldId) });
            toast.success(`Завантажено ${rows} спостережень`);
          } else if (update.state === "failed") {
            toast.error(update.error ?? "Не вдалося оновити дані");
          }
          setTimeout(() => {
            setProgress(null);
            setActiveFieldId(null);
          }, 1500);
        },
      });
    },
  });

  return { ...mutation, progress, activeFieldId };
}

export function useRequestHeatmap() {
  const qc = useQueryClient();
  const [progress, setProgress] = useState<JobUpdate | null>(null);

  const mutation = useMutation({
    mutationFn: ({
      fieldId,
      date,
      index,
    }: {
      fieldId: number;
      date: string;
      index: IndexName;
    }) => obsApi.requestHeatmap(fieldId, date, index),
    onSuccess: (data, vars) => {
      setProgress({ state: "queued" });
      subscribeToJob(data.job_id, {
        onUpdate: setProgress,
        onDone: (update) => {
          if (update.state === "done") {
            qc.invalidateQueries({ queryKey: KEY(vars.fieldId) });
            // Drop the cached blob so the PNG re-fetches now that it exists.
            qc.invalidateQueries({
              queryKey: HEATMAP_KEY(vars.fieldId, vars.date, vars.index),
            });
          } else if (update.state === "failed") {
            toast.error(update.error ?? "Не вдалося згенерувати heatmap");
          }
          setTimeout(() => setProgress(null), 1000);
        },
      });
    },
  });

  return { ...mutation, progress };
}

export function useQuotaPolling(intervalMs = 30_000) {
  const qc = useQueryClient();
  useEffect(() => {
    const id = window.setInterval(
      () => qc.invalidateQueries({ queryKey: ["quota"] }),
      intervalMs,
    );
    return () => window.clearInterval(id);
  }, [qc, intervalMs]);
}

/**
 * Track an already-enqueued observation job (e.g. the auto-fetch that the
 * backend kicks off on field creation). Mirrors the progress UX of the
 * "Refresh" button without re-enqueueing the job.
 *
 * Three completion-detection paths (we need all three because each plugs
 * a different hole):
 *  1. SSE stream — forwards future pub/sub events. Cheap and live, but
 *     Redis pub/sub doesn't replay, so events that fired before the
 *     subscription opened are lost.
 *  2. One-shot REST poll of /api/jobs/{id} on mount — catches the case
 *     where the RQ job already finished before SSE connected (common
 *     when the auto-fetch is faster than the field-list refetch).
 *  3. Slow polling fallback — every 5s, re-poll /api/jobs/{id} until a
 *     terminal state. Covers SSE drops, proxy buffering, and any edge
 *     case where #1 and #2 both miss the event. Costs at most a few
 *     extra HTTP requests per job; negligible.
 *
 * Whichever path detects the terminal state first wins; a flag prevents
 * the other two from firing duplicate toasts/invalidations.
 *
 * No `subscribedRef` guard here — React StrictMode mounts in dev fire
 * mount → cleanup → re-mount. A ref-based "already subscribed" check
 * would short-circuit the re-mount and leave the component with NO live
 * subscription. The useEffect dependency array is the only dedup we need.
 */
export function useTrackObservationJob(
  jobId: string | null | undefined,
  fieldId: number | null | undefined,
) {
  const qc = useQueryClient();
  const [progress, setProgress] = useState<JobUpdate | null>(null);

  useEffect(() => {
    if (!jobId || !fieldId) {
      setProgress(null);
      return;
    }
    setProgress({ state: "queued" });

    let cancelled = false;
    let finishedHandled = false;

    const handleDone = (state: "done" | "failed", data?: unknown, error?: string) => {
      if (finishedHandled || cancelled) return;
      finishedHandled = true;
      if (state === "done") {
        const d = data as { rows?: number } | undefined;
        const rows = d?.rows ?? 0;
        qc.invalidateQueries({ queryKey: KEY(fieldId) });
        toast.success(`Завантажено ${rows} спостережень`);
      } else {
        toast.error(error ?? "Не вдалося оновити дані");
      }
      setProgress({ state, ...(error ? { error } : {}) });
      setTimeout(() => setProgress(null), 1500);
    };

    const pollStatus = async () => {
      try {
        const resp = await api.get(`/api/jobs/${jobId}`);
        if (cancelled || finishedHandled) return;
        const state = resp.data?.state as string | undefined;
        if (state === "done") handleDone("done", resp.data?.result);
        else if (state === "failed")
          handleDone("failed", undefined, "Job failed");
      } catch {
        // 404 or transient — try again later.
      }
    };

    // Path 2: REST one-shot at mount.
    void pollStatus();

    // Path 3: slow polling backstop in case SSE drops or both other paths
    // miss the event. Stops itself once the job is handled.
    const pollIntervalId = window.setInterval(() => {
      if (cancelled || finishedHandled) {
        window.clearInterval(pollIntervalId);
        return;
      }
      void pollStatus();
    }, 5000);

    // Path 1: SSE for live updates.
    const close = subscribeToJob(jobId, {
      onUpdate: (u) => {
        if (!finishedHandled) setProgress(u);
      },
      onDone: (update) => {
        if (update.state === "done" || update.state === "failed") {
          handleDone(update.state, update.data, update.error);
        }
      },
    });

    return () => {
      cancelled = true;
      window.clearInterval(pollIntervalId);
      close();
    };
  }, [jobId, fieldId, qc]);

  return progress;
}

export { HEATMAP_KEY };
