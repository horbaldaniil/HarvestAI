import { useEffect, useState } from "react";

import * as authApi from "@/api/auth";
import { useAuthStore } from "@/stores/auth";

/**
 * One-shot startup effect that rehydrates the in-memory access token
 * from the long-lived `harvestai_refresh` httpOnly cookie the backend
 * issued at login time.
 *
 * Why this exists: `useAuthStore` is configured with `persist` that
 * only stores the `user` shape (not `accessToken`) — keeping the
 * access token out of localStorage limits XSS exposure. The trade-off
 * is that a page refresh wipes the in-memory token. Without a
 * rehydration step, `ProtectedRoute` redirects to `/login` even
 * though the refresh cookie in the browser is still valid.
 *
 * What it does:
 *   1. On mount, if no access token is in memory, POST /api/auth/refresh.
 *   2. The backend reads the httpOnly refresh cookie, mints a new
 *      access token + rotates the cookie, returns the access token.
 *   3. Follow up with GET /me to get the user shape.
 *   4. Push both into the Zustand store; the rest of the app proceeds
 *      as if the user was already logged in.
 *
 * What it does NOT do:
 *   - It doesn't periodically refresh. The existing axios response
 *     interceptor in `api/client.ts` handles mid-session 401s.
 *   - It doesn't redirect on failure. The `catch` block swallows the
 *     error and `ProtectedRoute` does the redirect to `/login` as
 *     designed when `accessToken` stays null.
 *
 * Returns `rehydrating` boolean — caller (main.tsx) shows a splash
 * spinner while true. Typical duration: 200–500 ms.
 */
export function useAuthRehydrate(): boolean {
  const accessToken = useAuthStore((s) => s.accessToken);
  const setAccessToken = useAuthStore((s) => s.setAccessToken);
  const setUser = useAuthStore((s) => s.setUser);

  // Only ever rehydrate ONCE per page-load. Initialise to true when we
  // have no token, false when we do (logged in already → skip the
  // round-trip entirely).
  const [rehydrating, setRehydrating] = useState(accessToken === null);

  useEffect(() => {
    let cancelled = false;
    if (accessToken !== null) {
      // Already authenticated — nothing to do.
      setRehydrating(false);
      return;
    }
    (async () => {
      try {
        const { access_token } = await authApi.refresh();
        if (cancelled) return;
        setAccessToken(access_token);
        // Fetch user shape so AccountCard / nav avatar / etc. have it
        // immediately instead of flickering through a loading state.
        const user = await authApi.me();
        if (cancelled) return;
        setUser(user);
      } catch {
        // No valid refresh cookie OR refresh-token revoked OR network
        // hiccup — silently fall through. ProtectedRoute will redirect
        // the user to /login when accessToken stays null.
      } finally {
        if (!cancelled) setRehydrating(false);
      }
    })();
    return () => {
      cancelled = true;
    };
    // Mount-only effect: we deliberately don't react to `accessToken`
    // changes after first run, because subsequent logins/logouts are
    // handled by the auth-mutation flow, not by this hook.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return rehydrating;
}
