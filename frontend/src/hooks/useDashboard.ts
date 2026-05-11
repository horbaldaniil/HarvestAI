import { useQuery } from "@tanstack/react-query";
import { getDashboard } from "@/api/dashboard";

export function useDashboard() {
  return useQuery({
    queryKey: ["dashboard"],
    queryFn: getDashboard,
    staleTime: 30 * 1000,
    refetchOnWindowFocus: true,
  });
}
