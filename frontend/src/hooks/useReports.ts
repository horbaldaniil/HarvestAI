import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import * as api from "@/api/reports";

const REPORTS_KEY = ["reports"] as const;

export function useReports() {
  return useQuery({
    queryKey: REPORTS_KEY,
    queryFn: api.listReports,
    staleTime: 30 * 1000,
  });
}

export function useRunBuilder() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (payload: api.BuilderPayload) => api.runReportBuilder(payload),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: REPORTS_KEY });
    },
    onError: () => {
      toast.error("Не вдалося згенерувати звіт.");
    },
  });
}

export function useDownloadSavedReport() {
  return useMutation({
    mutationFn: (id: number) => api.downloadSavedReport(id),
    onError: (err: unknown) => {
      const e = err as { response?: { status?: number } };
      if (e.response?.status === 410) {
        toast.error("Файл недоступний — згенеруйте знову.");
      } else {
        toast.error("Не вдалося завантажити звіт.");
      }
    },
  });
}

export function useDeleteReport() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.deleteReport(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: REPORTS_KEY });
      toast.success("Звіт видалено.");
    },
    onError: () => {
      toast.error("Не вдалося видалити звіт.");
    },
  });
}
