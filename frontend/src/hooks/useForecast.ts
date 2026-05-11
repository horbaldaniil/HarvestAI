import { useQuery } from "@tanstack/react-query";
import { fetchForecast } from "@/api/weather";

export function useForecast(fieldId: number | null | undefined, days = 14) {
  return useQuery({
    queryKey: ["forecast", fieldId, days],
    queryFn: () => fetchForecast(fieldId!, days),
    enabled: typeof fieldId === "number",
    staleTime: 5 * 60 * 1000,
  });
}
