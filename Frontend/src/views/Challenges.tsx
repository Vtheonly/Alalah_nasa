import { useEffect, useState } from "react";
import { AlertTriangle, CheckCircle2, Code2 } from "lucide-react";
import { api } from "../api";
import type { ChallengeDetail, ChallengeSummary, Region } from "../types";
import Chart from "../components/Chart";
import { KpiGrid } from "../components/Kpi";
import Icon from "../components/Icon";

export default function Challenges({ args }: { args: string[] }) {
  const [list, setList] = useState<ChallengeSummary[]>([]);
  const [regions, setRegions] = useState<Region[]>([]);
  const [iso, setIso] = useState<string | null>(null);
  const [detail, setDetail] = useState<ChallengeDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const id = args[0] ?? list[0]?.id;

  useEffect(() => {
    api.challenges().then(setList).catch(() => undefined);
    api.regions().then(setRegions).catch(() => undefined);
  }, []);
  useEffect(() => {
    if (!id) return;
    const ac = new AbortController();
    setLoading(true);
    api.challenge(id, iso, ac.signal).then((d) => { setDetail(d); setLoading(false); }).catch(() => undefined);
    return () => ac.abort();
  }, [id, iso]);

  const idx = list.findIndex((c) => c.id === id);
  return (
    <div className="stack-lg">
      <div className="page-head">
        <div>
          <h1>Challenges &amp; solutions</h1>
          <p className="muted">Each scientific or technical obstacle, how the engine solves it, and the evidence computed live from the data.</p>
        </div>
        <div className="chips">
          <button className={!iso ? "chip active" : "chip"} onClick={() => setIso(null)}>All countries</button>
          {regions.map((r) => <button key={r.iso} className={iso === r.iso ? "chip active" : "chip"} onClick={() => setIso(r.iso)}>{r.iso}</button>)}
        </div>
      </div>
      <div className="challenges">
        <nav className="challenge-nav">
          {list.map((c, i) => (
            <a key={c.id} href={`#/challenges/${c.id}`} className={c.id === id ? "cnav active" : "cnav"}>
              <span className="cnav-n">{String(i + 1).padStart(2, "0")}</span>
              <span className="icon-bubble"><Icon name={c.icon} size={15} /></span>
              <span>{c.title}</span>
            </a>
          ))}
        </nav>
        <article className={`card challenge ${loading ? "loading" : ""}`}>
          {detail ? (
            <>
              <div className="challenge-head">
                <span className="icon-bubble big"><Icon name={detail.icon} size={22} /></span>
                <div>
                  <div className="eyebrow">Challenge {idx + 1} of {list.length}</div>
                  <h2>{detail.title}</h2>
                </div>
              </div>
              <div className="ps-grid">
                <div className="ps problem"><AlertTriangle size={16} /><div><b>The problem</b><p>{detail.problem}</p></div></div>
                <div className="ps solution"><CheckCircle2 size={16} /><div><b>How it is solved</b><p>{detail.solution}</p></div></div>
              </div>
              <KpiGrid items={detail.kpis} />
              <div className={detail.charts.length > 1 ? "grid-2" : ""}>
                {detail.charts.map((c) => <Chart key={c.title} spec={c} height={300} />)}
              </div>
              {detail.charts.find((c) => c.h3_index) && (
                <a className="link" href={`#/trace/${detail.charts.find((c) => c.h3_index)!.h3_index}`}>Trace this hexagon to its satellite detections →</a>
              )}
              <div className="modules"><Code2 size={14} /> Implemented in {detail.pipeline.map((p) => <code key={p}>{p}</code>)}</div>
            </>
          ) : <div className="skeleton" style={{ height: 480 }} />}
        </article>
      </div>
    </div>
  );
}
