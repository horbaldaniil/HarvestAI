import { useQuery } from "@tanstack/react-query";

import { getDashboard, type DashboardFilters } from "@/api/dashboard";

export function useDashboard(filters: DashboardFilters = {}) {
  return useQuery({
    queryKey: ["dashboard", filters.crops ?? []],
    queryFn: () => getDashboard(filters),
    staleTime: 30 * 1000,
    refetchOnWindowFocus: true,
  });
}
