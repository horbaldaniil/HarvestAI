import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useAuthStore } from "@/stores/auth";
import * as authApi from "@/api/auth";

export function useCurrentUser() {
  const token = useAuthStore((s) => s.accessToken);
  return useQuery({
    queryKey: ["currentUser"],
    queryFn: authApi.me,
    enabled: !!token,
    staleTime: 5 * 60 * 1000,
  });
}

export function useLogin() {
  const setAccessToken = useAuthStore((s) => s.setAccessToken);
  const qc = useQueryClient();
  return useMutation({
    mutationFn: authApi.login,
    onSuccess: async (data) => {
      setAccessToken(data.access_token);
      await qc.invalidateQueries({ queryKey: ["currentUser"] });
    },
  });
}

export function useRegister() {
  return useMutation({
    mutationFn: authApi.register,
  });
}

export function useLogout() {
  const clear = useAuthStore((s) => s.clear);
  const qc = useQueryClient();
  return useMutation({
    mutationFn: authApi.logout,
    onSettled: () => {
      clear();
      qc.clear();
    },
  });
}
