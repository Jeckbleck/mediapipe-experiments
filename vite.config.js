import { defineConfig } from "vite";

export default defineConfig({
  root: ".",
  // GitHub Pages serves from /<repo>/; Docker and local dev serve from /
  base: process.env.BASE_PATH ?? "/",
  server: {
    port: 5173,
  },
});
