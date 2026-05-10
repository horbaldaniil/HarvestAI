import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { User } from "@/api/auth";

interface AuthState {
  accessToken: string | null;
  user: User | null;
  setAccessToken: (token: string | null) => void;
  setUser: (user: User | null) => void;
  clear: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      accessToken: null,
      user: null,
      setAccessToken: (token) => set({ accessToken: token }),
      setUser: (user) => set({ user }),
      clear: () => set({ accessToken: null, user: null }),
    }),
    {
      name: "harvestai-auth",
      // Don't persist access token — only the user shape for hydration UX.
      partialize: (state) => ({ user: state.user }),
    }
  )
);
