import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The Python API runs on :8000; proxy /api to it so the browser talks to a
// single origin and no CORS config is needed in normal use.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
});
