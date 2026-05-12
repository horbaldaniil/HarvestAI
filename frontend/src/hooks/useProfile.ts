import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import * as api from "@/api/settings";

const KEY = ["settings", "profile"] as const;

/**
 * Read the current user's profile (full_name + farm metadata).
 * Flat shape; the backend hides the split between `users.full_name`
 * (table column) and `settings_json["profile"]` (JSONB).
 */
export function useProfile() {
  return useQuery({
    queryKey: KEY,
    queryFn: api.getProfile,
    staleTime: 5 * 60 * 1000,
  });
}

/**
 * Save partial profile updates. Empty strings are treated as "clear
 * this field" by the backend, so a user who wipes their phone-number
 * input and saves will see it removed from `settings_json`.
 *
 * On success we also nudge the `currentUser` query — AccountCard
 * pulls `full_name` through that path and would otherwise stay
 * stale until the next page refresh.
 */
export function useUpdateProfile() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (payload: api.ProfileUpdate) => api.updateProfile(payload),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: KEY });
      qc.invalidateQueries({ queryKey: ["currentUser"] });
      toast.success("Профіль оновлено");
    },
    onError: () => {
      toast.error("Не вдалося зберегти профіль");
    },
  });
}
