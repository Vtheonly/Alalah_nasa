# Fire Harmonizer — web frontend

React + Vite + ECharts + MapLibre. The frontend contains **no data and no
provider logic**: every number, chart and label comes from the FastAPI
backend (`src/webapp/server.py`), which decides server-side whether it serves
the real harmonization engine output or the clearly labelled synthetic
demonstration store. The UI is identical in both cases and shows the
provenance the backend reports on every page.

## Pages

| Page | What it demonstrates |
|---|---|
| Overview | Satellites behind the record, the processing pipeline, validation summary |
| Explorer | H3 map (fire power / anomaly / uncertainty / evidence), month player, burning calendar, time series (raw vs harmonized) |
| Explorer → Replay | Pick a month on the burning calendar and replay it **day by day** or **overpass by overpass (hour)**: hexagons glow with the harmonized daily fire power and fade (heat trail), raw detections flash coloured by satellite, activity strip to scrub |
| Challenges & solutions | 8 scientific/technical challenges, each with problem, solution, live evidence and the implementing modules |
| Validation | Every check: question, threshold, result, evidence chart; the gate the server uses to call a run "validated" |
| Insights | Trend (Theil–Sen, Mann–Kendall), fire season, most extreme months with attribution, country comparison |
| Traceability | One hexagon-month traced to every satellite detection, with weights, flags and a reconciliation check |

## Run

```bash
cd Frontend && npm install && npm run build        # or: make web
cd .. && make serve                                # http://localhost:8000
```

Development with hot reload: `make serve` in one terminal, `npm run dev` in
`Frontend/` in another (Vite proxies `/api` to `http://127.0.0.1:8000`, or
`VITE_API_PROXY_TARGET`).

## Choosing the data (backend only)

| Variable | Values | Default |
|---|---|---|
| `FH_DATA_PROVIDER` | `auto`, `real`, `synthetic` | `auto` |
| `FH_SYNTHETIC_FALLBACK` | `true`, `false` | `true` |
| `FIRE_HARMONIZED_DB` | real engine output | `output/fire_harmonized.duckdb` |
| `FH_SYNTHETIC_DB` | synthetic store (built on first use, ~15 s) | `output/synthetic_demo.duckdb` |

`auto`/`real` serve the real output only if it passes the critical validation
checks; otherwise the synthetic store (fallback on) or the real output labelled
*not validated* (fallback off). `make demo` forces synthetic data.
