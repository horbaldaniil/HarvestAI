import { useEffect } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import type { IndexName } from "@/api/observations";
import { HEATMAP_KEY } from "./useObservations";

/**
 * Fetch a heatmap PNG via the authenticated axios client and expose it as
 * an object URL that Leaflet's <ImageOverlay /> can use.
 *
 * Why this exists: <ImageOverlay url={...} /> performs a plain browser
 * GET (via an <img> tag) which can't carry JWT Authorization headers.
 * The backend endpoint is auth-gated, so a direct URL load → 401. We
 * fetch via axios (which has the JWT interceptor), receive a Blob,
 * wrap it in URL.createObjectURL(), and pass that as the layer's URL.
 *
 * The object URL is revoked when the component unmounts or the cached
 * data changes, so we don't leak memory.
 *
 * Retry behaviour: when the user clicks a chart point, the RQ job that
 * renders the PNG is dispatched simultaneously. The first PNG GET will
 * 404 until the worker finishes. We rely on React Query's invalidation
 * (fired by useRequestHeatmap's SSE onDone) to refetch — no need for
 * automatic retries here.
 */
export function useHeatmapBlob(
  fieldId: number,
  date: string | null,
  index: IndexName,
) {
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: date ? HEATMAP_KEY(fieldId, date, index) : ["heatmap-blob", "disabled"],
    queryFn: async () => {
      if (!date) return null;
      const { data } = await api.get<Blob>(
        `/api/fields/${fieldId}/heatmaps/${date}_${index}.png`,
        { responseType: "blob" },
      );
      return URL.createObjectURL(data);
    },
    enabled: !!date,
    retry: false,
    staleTime: Infinity,
    gcTime: 5 * 60 * 1000,
  });

  // Revoke the URL when the data is replaced or component unmounts.
  useEffect(() => {
    const url = query.data;
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  }, [query.data]);

  // Also clean up any stale cache for THIS (field, date, index) on unmount,
  // so a re-mount with the same params re-fetches a fresh URL.
  useEffect(() => {
    if (!date) return;
    return () => {
      qc.removeQueries({ queryKey: HEATMAP_KEY(fieldId, date, index) });
    };
  }, [fieldId, date, index, qc]);

  return query;
}
