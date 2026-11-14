# Ember Atlas frontend

React + Vite dashboard for the Fire Harmonizer FastAPI service.

## Run locally

Start the API from the repository root:

```powershell
python -m uvicorn src.webapp.server:app --reload
```

In another terminal:

```powershell
cd Frontend
npm install
npm run dev
```

Open the Vite URL printed in the terminal. The development server proxies
`/api` requests to `http://127.0.0.1:8000`; set `VITE_API_PROXY_TARGET` to use
a different API host.

The calendar, time series, and segmented map are fetched from `/api/calendar`,
`/api/timeseries`, and `/api/map_hex`. If the API is unavailable or has no
records for the selected area/month, the affected view uses deterministic
demonstration data and labels it as such. CARTO's OpenStreetMap-based basemap
requires an internet connection.

## Build

```powershell
npm run build
```
