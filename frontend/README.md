# RagBench Frontend

Minimal React + Tailwind dashboard for the RagBench project.

## Setup

```bash
cd frontend
npm install
npm run dev
```

Opens at http://localhost:5173. The dev server proxies `/api` requests
to the FastAPI backend at `http://localhost:8000`, so you can run the
backend and frontend side by side during development.

## Stack

- **Vite** — build tool and dev server
- **React 18** — UI library
- **React Router** — client-side routing
- **Tailwind CSS** — utility-first styling

## Structure

```
frontend/
├── index.html              # HTML entry + Google Fonts (Inter)
├── vite.config.js          # Vite + /api proxy to FastAPI
├── tailwind.config.js      # Custom accent color, font stack
├── src/
│   ├── main.jsx            # React entry point
│   ├── App.jsx             # Route definitions
│   ├── index.css           # Tailwind directives + base styles
│   ├── components/
│   │   ├── Layout.jsx      # Page shell (header + sidebar + main)
│   │   ├── Header.jsx      # Top bar with title + GitHub link
│   │   └── Sidebar.jsx     # Left nav with active states
│   └── pages/
│       ├── Datasets.jsx
│       ├── Runs.jsx
│       ├── Metrics.jsx
│       └── About.jsx
```

## Design Philosophy

Academic, clean, minimal. Inspired by arXiv, Notion, and simple research
dashboards. Neutral palette (white, slate, soft blue accent), generous
whitespace, readable type.
