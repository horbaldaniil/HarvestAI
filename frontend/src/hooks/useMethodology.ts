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

/** Phase 4 leaderboard: 1 row per (crop, family) with headline metrics. */
export function useLeaderboard() {
  return useQuery({
    queryKey: ["methodology", "leaderboard"] as const,
    queryFn: api.getLeaderboard,
    staleTime: 5 * 60 * 1000,
  });
}

/** Phase 4 full evaluation: SHAP, residuals, learning curves, etc. */
export function useEvaluationV3() {
  return useQuery({
    queryKey: ["methodology", "evaluation_v3"] as const,
    queryFn: api.getEvaluationV3,
    staleTime: 5 * 60 * 1000,
  });
}
