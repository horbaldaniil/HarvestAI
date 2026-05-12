import { useQuery } from "@tanstack/react-query";

import * as api from "@/api/weather";

/**
 * Detailed weather payload for the dedicated /weather page.
 * Includes 7-day forecast + rule-based advisories from the backend.
 */
export function useWeatherDetail(fieldId: number | null, days = 14) {
  return useQuery({
    queryKey: ["weather", "detail", fieldId, days],
    queryFn: () => api.getWeatherDetail(fieldId!, days),
    enabled: typeof fieldId === "number",
    staleTime: 10 * 60 * 1000,
  });
}
