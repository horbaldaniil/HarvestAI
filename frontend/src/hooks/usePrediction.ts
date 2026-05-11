import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import * as api from "@/api/predictions";
import { subscribeToJob, type JobUpdate } from "@/lib/sse";

const KEY = (fieldId: number) => ["prediction", fieldId] as const;

export function useLatestPrediction(fieldId: number | null | undefined) {
  return useQuery({
    queryKey: fieldId ? KEY(fieldId) : ["prediction", "disabled"],
    queryFn: () => api.getLatestPrediction(fieldId!),
    enabled: typeof fieldId === "number",
    staleTime: 30 * 1000,
  });
}

export function useRecomputePrediction() {
  const qc = useQueryClient();
  const [progress, setProgress] = useState<JobUpdate | null>(null);

  const mutation = useMutation({
    mutationFn: (fieldId: number) => api.recomputePrediction(fieldId),
    onSuccess: (data, fieldId) => {
      setProgress({ state: "queued" });
      subscribeToJob(data.job_id, {
        onUpdate: setProgress,
        onDone: (update) => {
          if (update.state === "done") {
            qc.invalidateQueries({ queryKey: KEY(fieldId) });
            toast.success("Прогноз оновлено");
          } else if (update.state === "failed") {
            toast.error(update.error ?? "Не вдалося оновити прогноз");
          }
          setTimeout(() => setProgress(null), 1500);
        },
      });
    },
  });

  return { ...mutation, progress };
}
