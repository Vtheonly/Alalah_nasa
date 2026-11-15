import { lazy, Suspense, useEffect, useState } from "react";
import { Activity, BarChart3, CheckCircle2, Compass, Flame, Globe2, Microscope, Puzzle } from "lucide-react";
import { api } from "./api";
import type { Status } from "./types";
import { ProvenanceBanner, ProvenancePill } from "./components/Provenance";
import Overview from "./views/Overview";
import Explorer from "./views/Explorer";
import Challenges from "./views/Challenges";
import ValidationView from "./views/Validation";
import InsightsView from "./views/Insights";
import Trace from "./views/Trace";

const GlobeView = lazy(() => import("./views/GlobeView"));   // three.js only loads here

export type Route = { page: string; args: string[] };

function parseHash(): Route {
  const parts = window.location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  return { page: parts[0] ?? "overview", args: parts.slice(1) };
}

export function go(path: string) {
  window.location.hash = `#/${path}`;
}

const NAV = [
  { id: "overview", label: "Overview", icon: Compass },
  { id: "globe", label: "3D globe", icon: Globe2 },
  { id: "explorer", label: "Explorer", icon: Activity },
  { id: "challenges", label: "Challenges & solutions", icon: Puzzle },
  { id: "validation", label: "Validation", icon: CheckCircle2 },
  { id: "insights", label: "Insights", icon: BarChart3 },
  { id: "trace", label: "Traceability", icon: Microscope },
];

export default function App() {
  const [route, setRoute] = useState<Route>(parseHash());
  const [status, setStatus] = useState<Status | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const on = () => { setRoute(parseHash()); window.scrollTo({ top: 0 }); };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);

  useEffect(() => {
    api.status().then(setStatus).catch((e: Error) => setError(e.message));
  }, []);

  const page = route.page;
  return (
    <div className="shell">
      <header className="topbar">
        <a className="brand" href="#/overview">
          <span className="brand-mark"><Flame size={18} /></span>
          <span className="brand-text"><b>Fire Harmonizer</b><small>MODIS + VIIRS · 2000–2026</small></span>
        </a>
        <nav className="tabs">
          {NAV.map(({ id, label, icon: Icon }) => (
            <a key={id} href={`#/${id}`} className={page === id ? "tab active" : "tab"}>
              <Icon size={15} /><span>{label}</span>
            </a>
          ))}
        </nav>
        <ProvenancePill p={status?.provenance} />
      </header>
      <ProvenanceBanner status={status} />
      {error && <div className="banner banner-unvalidated">API unreachable: {error}</div>}
      <main className="page">
        {page === "overview" && <Overview status={status} />}
        {page === "globe" && <Suspense fallback={<div className="skeleton" style={{ height: 640 }} />}><GlobeView /></Suspense>}
        {page === "explorer" && <Explorer args={route.args} />}
        {page === "challenges" && <Challenges args={route.args} />}
        {page === "validation" && <ValidationView />}
        {page === "insights" && <InsightsView args={route.args} />}
        {page === "trace" && <Trace args={route.args} />}
      </main>
      <footer className="footer">
        <span>NASA Space Apps · Harmonized MODIS–VIIRS active fire record on an H3 grid</span>
        <span>Data source shown on every page: {status ? status.provenance.label : "…"}</span>
      </footer>
    </div>
  );
}
