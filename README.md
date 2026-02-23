# MediaPipe Experiments

Facial expression, gesture recognition, face swap, and AI photobooth demos.

## Quick start

```bash
npm install
npm run dev
```

Open http://localhost:5173

## Python backend (recommended for face swap & AR)

For more reliable face swap and AR props, run the Python backend:

```bash
# Terminal 1: install and run backend
npm run backend:install
npm run backend

# Terminal 2: run frontend
npm run dev
```

Then enable **"Use Python backend"** in the Face Mesh, Face Swap Single, or Face Swap Multi pages.

See [backend/README.md](backend/README.md) for details.
