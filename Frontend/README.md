# Fire Harmonizer frontend

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

## Globe and H3 fire animation

The map opens on a global globe. Choose a study area from the sidebar or search
to fly into it; use the globe button to return to the global view. Regional map
features use the H3 polygons returned by `/api/map_hex`. Their extrusion height
is scaled from each cell's harmonized median fire radiative power, while the
color control can display FRP, anomaly, or uncertainty.

The timeline's Play control advances through the monthly harmonized output.
H3 cells entering or leaving the active month fade in/out, and their heat-based
extrusions interpolate between monthly values. When the API is unavailable,
the clearly labelled demonstration layer is generated from real H3 cell
boundaries at resolution 6; it is not presented as observed fire data.

## Build

```powershell
npm run build
```
