import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { RouterProvider } from "react-router-dom";
import { Loader2 } from "lucide-react";
import { Toaster } from "sonner";

import "./styles/globals.css";
import "./lib/i18n";
import { router } from "./router";
import { TooltipProvider } from "./components/ui/tooltip";
import { useAuthRehydrate } from "./hooks/useAuthRehydrate";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30 * 1000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});


/**
 * Auth-rehydration gate that wraps the router. On page-refresh, the
 * in-memory access token is empty (the Zustand `persist` partializer
 * keeps it out of localStorage by design) — so we trade the surviving
 * `harvestai_refresh` httpOnly cookie for a fresh access token before
 * letting ProtectedRoute evaluate. Without this gate, every refresh
 * would bounce the user to /login despite a valid session.
 *
 * UX: fullscreen spinner for ~200-500 ms. The hook itself swallows
 * any failure (expired cookie, network down) silently — the router
 * then proceeds and ProtectedRoute does its normal redirect.
 */
function AppRoot() {
  const rehydrating = useAuthRehydrate();
  if (rehydrating) {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-background">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }
  return <RouterProvider router={router} />;
}


createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      {/* delayDuration=200 keeps tooltips snappy without firing on the
          briefest mouse-overs while panning Leaflet. */}
      <TooltipProvider delayDuration={200}>
        <AppRoot />
      </TooltipProvider>
      <Toaster position="top-right" richColors />
    </QueryClientProvider>
  </StrictMode>
);
