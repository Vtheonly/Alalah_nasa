import { useEffect, useState } from "react";
import { CheckCircle2, CircleDashed, ShieldCheck, ShieldX, XCircle } from "lucide-react";
import { api } from "../api";
import type { Validation } from "../types";
import Chart from "../components/Chart";
import { fmt } from "../format";

export default function ValidationView() {
  const [v, setV] = useState<Validation | null>(null);
  useEffect(() => { api.validation().then(setV).catch(() => undefined); }, []);
  if (!v) return <div className="skeleton" style={{ height: 500 }} />;
  const synthetic = v.provenance.is_synthetic;
  return (
    <div className="stack-lg">
      <div className="page-head">
        <div>
          <h1>Validation</h1>
          <p className="muted">Does harmonization actually work? Every check states its question, threshold, result and evidence.</p>
        </div>
      </div>
      <section className={`gate ${v.gate.passed ? "ok" : "ko"}`}>
        {v.gate.passed ? <ShieldCheck size={28} /> : <ShieldX size={28} />}
        <div>
          <b>{v.gate.passed ? "All critical checks pass" : `Critical checks failing: ${v.gate.failed.join(", ") || "no diagnostics"}`}</b>
          <p>
            The server only labels a real engine run <i>validated</i> when these critical checks pass: {v.gate.critical.join(", ")}.
            {synthetic && " These results come from the synthetic demonstration data: they show how validation is presented, not evidence about real fires."}
          </p>
        </div>
      </section>
      <div className="checks">
        {v.checks.map((c) => (
          <article key={c.id} className={`card check ${c.status}`}>
            <header>
              {c.status === "pass" ? <CheckCircle2 size={18} /> : c.status === "fail" ? <XCircle size={18} /> : <CircleDashed size={18} />}
              <h3>{c.title}</h3>
              {c.critical && <span className="crit">critical</span>}
              <span className={`status ${c.status}`}>{c.status}</span>
            </header>
            <p className="muted">{c.question}</p>
            <div className="check-metric">
              <div><small>Result</small><b>{fmt(c.value, c.format)}</b></div>
              <div><small>Pass if</small><span>{c.threshold}</span></div>
            </div>
            {c.evidence && <Chart spec={c.evidence} height={220} />}
            {c.detail && (
              <details>
                <summary>Supporting numbers</summary>
                <table className="kv">
                  <tbody>
                    {Object.entries(c.detail).map(([k, val]) => (
                      <tr key={k}><td>{k.replace(/_/g, " ")}</td><td>{typeof val === "number" ? fmt(val, Math.abs(val) < 0.01 ? "p" : "num") : String(val)}</td></tr>
                    ))}
                  </tbody>
                </table>
              </details>
            )}
            {c.status === "unavailable" && <p className="muted small">Not computed for this dataset (required diagnostics absent).</p>}
          </article>
        ))}
      </div>
    </div>
  );
}
