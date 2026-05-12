import { useQuery } from "@tanstack/react-query";

import * as api from "@/api/methodology";

export function useMethodologyOverview() {
  return useQuery({
    queryKey: ["methodology", "overview"] as const,
    queryFn: api.getMethodologyOverview,
    staleTime: 5 * 60 * 1000,
  });
}

export function useMethodologyMetrics() {
  return useQuery({
    queryKey: ["methodology", "metrics"] as const,
    queryFn: api.getMethodologyMetrics,
    staleTime: 5 * 60 * 1000,
  });
}

export function useCoverage() {
  return useQuery({
    queryKey: ["methodology", "coverage"] as const,
    queryFn: api.getCoverage,
    staleTime: 5 * 60 * 1000,
  });
}
