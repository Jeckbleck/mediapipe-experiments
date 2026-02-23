# Python Backend

Face swap, AR props, and landmarks processing using MediaPipe + OpenCV.

## Setup

```bash
cd backend
pip install -r requirements.txt
```

Or from project root:

```bash
npm run backend:install
```

## Run

```bash
cd backend
python run.py
```

Or from project root:

```bash
npm run backend
```

The backend runs on **http://127.0.0.1:8000**. The Vite dev server proxies `/api` to it.

## API

- `GET /api/health` – health check
- `POST /api/process` – process a frame
  - Form fields: `frame` (base64), `mode`, `reference_image` (optional), `prop` (for ar-props)
  - Modes: `landmarks`, `face-swap`, `face-swap-multi`, `ar-props`
  - Returns: `{ success, image?, error? }`

## Usage

1. Start the backend: `npm run backend`
2. Start the frontend: `npm run dev`
3. Open Face Mesh, Face Swap Single, or Face Swap Multi
4. Enable "Use Python backend" in the sidebar
