import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import * as api from "@/api/settings";

const KEY = ["settings", "crop-prices"] as const;

export function useCropPrices() {
  return useQuery({
    queryKey: KEY,
    queryFn: api.getCropPrices,
    staleTime: 5 * 60 * 1000,
  });
}

export function useUpdateCropPrices() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (payload: api.CropPricesUpdate) => api.updateCropPrices(payload),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: KEY });
      // Income card on the dashboard reads the cached prices through this
      // query — also refetch the dashboard so the projection updates.
      qc.invalidateQueries({ queryKey: ["dashboard"] });
      toast.success("Ціни оновлено");
    },
    onError: () => {
      toast.error("Не вдалося зберегти ціни");
    },
  });
}
