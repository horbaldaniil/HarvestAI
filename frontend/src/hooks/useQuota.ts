import { useQuery } from "@tanstack/react-query";
import { getQuota } from "@/api/quota";

export function useQuota() {
  return useQuery({
    queryKey: ["quota"],
    queryFn: getQuota,
    staleTime: 30 * 1000,
  });
}
