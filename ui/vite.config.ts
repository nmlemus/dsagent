import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The production build is served by `dsagent serve` from `ui/dist/` at `/`.
// In dev, Vite (:5173) proxies the API surfaces to the running server (:8000)
// so there is no CORS story and the SPA talks to the real backend.
const API = process.env.DSAGENT_API ?? "http://localhost:8000";
const proxy = Object.fromEntries(
  ["/agent", "/projects", "/sessions", "/runs", "/cartridges", "/chart-theme", "/a2ui"].map((p) => [
    p,
    { target: API, changeOrigin: true },
  ]),
);

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy },
  build: { outDir: "dist", sourcemap: false },
});
