import { useEffect, useState } from "react";
import { ArrowRight, CheckCircle2, XCircle } from "lucide-react";
import { api } from "../api";
import type { ChallengeSummary, Overview as OV, Status, Validation } from "../types";
import { PLATFORM_COLORS } from "../colors";
import { compact, fmt } from "../format";
import Icon from "../components/Icon";
import { Stat } from "../components/Kpi";

const PIPELINE = [
  { t: "Raw detections", d: "Every MODIS & VIIRS fire pixel, 5 satellites", c: "constellation" },
  { t: "Hygiene", d: "Flares, SAMA, duplicates removed", c: "non-fire-sources" },
  { t: "Geometry & time", d: "Pixel size, view angle, local solar time", c: "overpass-timing" },
  { t: "Sensor operators", d: "Calibrated on the 2012–21 overlap", c: "resolution" },
  { t: "Bayesian fusion", d: "GPU solver, day & night states, clouds", c: "missing-observations" },
  { t: "Honest uncertainty", d: "Conformal hold-out calibration", c: "uncertainty" },
];

function Timeline({ ov, events }: { ov: OV; events: Status["constellation_events"] }) {
  const t0 = new Date("2000-11-01").getTime();
  const t1 = new Date("2026-11-14").getTime();
  const pos = (d: string) => `${(100 * (new Date(d).getTime() - t0)) / (t1 - t0)}%`;
  return (
    <div className="timeline">
      {ov.platforms.map((p) => (
        <div key={p.platform} className="tl-row">
          <span className="tl-name">{p.platform}<small>{p.sensor}</small></span>
          <div className="tl-track">
            <div className="tl-bar" style={{ left: pos(p.first), width: `calc(${pos(p.last)} - ${pos(p.first)})`, background: PLATFORM_COLORS[p.platform] }}>
              <span>{compact(p.n_detections)} detections</span>
            </div>
          </div>
        </div>
      ))}
      <div className="tl-row tl-axis">
        <span className="tl-name" />
        <div className="tl-track">
          {[2002, 2006, 2010, 2014, 2018, 2022, 2026].map((y) => (
            <span key={y} className="tl-tick" style={{ left: pos(`${y}-01-01`) }}>{y}</span>
          ))}
          {events.map((e) => (
            <span key={e.month} className="tl-event" style={{ left: pos(`${e.month}-01`) }} title={e.label} />
          ))}
        </div>
      </div>
    </div>
  );
}

export default function Overview({ status }: { status: Status | null }) {
  const [ov, setOv] = useState<OV | null>(null);
  const [ch, setCh] = useState<ChallengeSummary[]>([]);
  const [val, setVal] = useState<Validation | null>(null);
  useEffect(() => {
    api.overview().then(setOv).catch(() => undefined);
    api.challenges().then(setCh).catch(() => undefined);
    api.validation().then(setVal).catch(() => undefined);
  }, []);

  const passed = val?.checks.filter((c) => c.status === "pass").length ?? 0;
  const avail = val?.checks.filter((c) => c.status !== "unavailable").length ?? 0;
  return (
    <div className="stack-lg">
      <section className="hero">
        <div className="hero-copy">
          <div className="eyebrow">One continuous fire record from five satellites</div>
          <h1>Turning two generations of fire satellites into <span className="grad">one consistent 26-year record</span></h1>
          <p>MODIS (1 km, since 2000) and VIIRS (375 m, since 2012) see fire differently. Fire Harmonizer fuses every detection into a single, uncertainty-aware estimate of fire energy on a global hexagon grid, and shows its evidence.</p>
          <div className="hero-actions">
            <a className="btn primary" href="#/explorer">Explore the map <ArrowRight size={15} /></a>
            <a className="btn" href="#/globe">Watch the satellites in 3D</a>
            <a className="btn" href="#/challenges">See how each problem is solved</a>
          </div>
        </div>
        <div className="hero-stats">
          <Stat value={ov ? compact(ov.n_detections ?? 0) : "…"} label="satellite detections" />
          <Stat value={ov ? compact(ov.n_cells, 0) : "…"} label="H3 hexagons (36 km²)" />
          <Stat value={ov ? `${ov.first_year}–${ov.last_year}` : "…"} label="monthly record" />
          <Stat value={val ? `${passed}/${avail}` : "…"} label="validation checks passed" accent={val?.gate.passed ? "#34d399" : "#f87171"} />
        </div>
      </section>

      <section className="card">
        <div className="card-head"><h2>Satellites behind the record</h2><span className="muted">dashed marks: launches, orbit drift, S-NPP end</span></div>
        {ov && status ? <Timeline ov={ov} events={status.constellation_events} /> : <div className="skeleton" style={{ height: 180 }} />}
      </section>

      <section>
        <div className="section-head"><h2>From raw pixels to a harmonized record</h2><a href="#/challenges" className="link">All challenges <ArrowRight size={14} /></a></div>
        <div className="pipeline">
          {PIPELINE.map((s, i) => (
            <a key={s.t} className="pipe-step" href={`#/challenges/${s.c}`}>
              <span className="pipe-n">{i + 1}</span>
              <b>{s.t}</b>
              <small>{s.d}</small>
            </a>
          ))}
        </div>
      </section>

      <section className="grid-2">
        <div className="card">
          <div className="card-head"><h2>Challenges solved</h2></div>
          <div className="challenge-list">
            {ch.map((c) => (
              <a key={c.id} className="challenge-row" href={`#/challenges/${c.id}`}>
                <span className="icon-bubble"><Icon name={c.icon} size={16} /></span>
                <span><b>{c.title}</b><small>{c.problem}</small></span>
                <ArrowRight size={14} className="muted" />
              </a>
            ))}
          </div>
        </div>
        <div className="card">
          <div className="card-head"><h2>Does it work?</h2><a href="#/validation" className="link">Evidence <ArrowRight size={14} /></a></div>
          <div className="check-list">
            {val?.checks.map((c) => (
              <div key={c.id} className={`check-row ${c.status}`}>
                {c.status === "pass" ? <CheckCircle2 size={16} /> : c.status === "fail" ? <XCircle size={16} /> : <span className="dot" />}
                <span>{c.title}{c.critical && <em className="crit">critical</em>}</span>
                <b>{fmt(c.value, c.format)}</b>
              </div>
            ))}
          </div>
          {val && <p className="muted small">Reference: {val.reference === "synthetic_truth" ? "the known synthetic truth (demonstration)" : "the full multi-sensor product and independent burned area"}.</p>}
        </div>
      </section>

      <section>
        <div className="section-head"><h2>Study countries</h2></div>
        <div className="country-grid">
          {ov?.countries.map((c) => (
            <a key={c.iso} className="country-card" href={`#/explorer/${c.iso}`}>
              <span className="iso">{c.iso}</span>
              <b>{c.name}</b>
              <span className="muted">{compact(c.n_cells, 0)} hexagons · {compact(c.fre_tj_per_year)} TJ/yr</span>
            </a>
          ))}
        </div>
      </section>
    </div>
  );
}
