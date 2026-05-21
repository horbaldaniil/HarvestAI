import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        // Explicit IPv4 — Node 17+ resolves "localhost" to IPv6 (::1) first,
        // but uvicorn binds only to IPv4 (127.0.0.1) by default. Without
        // pinning, macOS dev sessions get "ECONNREFUSED ::1:8000" on
        // every API call. Hard-coding 127.0.0.1 sidesteps the DNS
        // dual-stack guessing entirely.
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
