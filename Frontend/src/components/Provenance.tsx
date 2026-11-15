import { FlaskConical, ShieldAlert, ShieldCheck } from "lucide-react";
import type { Provenance, Status } from "../types";

export function provenanceTone(p?: Provenance): "synthetic" | "validated" | "unvalidated" {
  if (!p || p.is_synthetic) return "synthetic";
  return p.validated ? "validated" : "unvalidated";
}

export function ProvenancePill({ p }: { p?: Provenance }) {
  const tone = provenanceTone(p);
  const Icon = tone === "validated" ? ShieldCheck : tone === "synthetic" ? FlaskConical : ShieldAlert;
  const text = tone === "validated" ? "Real engine · validated" : tone === "synthetic" ? "Synthetic demo data" : "Real engine · not validated";
  return <span className={`pill pill-${tone}`}><Icon size={13} /> {text}</span>;
}

export function ProvenanceBanner({ status }: { status: Status | null }) {
  if (!status) return null;
  const p = status.provenance;
  const tone = provenanceTone(p);
  if (tone === "validated") {
    return (
      <div className="banner banner-validated">
        <ShieldCheck size={16} />
        <span><b>Real harmonization output.</b> Passed all critical validation checks ({status.validation_gate.critical.length}/{status.validation_gate.critical.length}).</span>
      </div>
    );
  }
  return (
    <div className={`banner banner-${tone}`}>
      {tone === "synthetic" ? <FlaskConical size={16} /> : <ShieldAlert size={16} />}
      <span>
        {tone === "synthetic"
          ? <><b>Synthetic demonstration data.</b> Simulated values that follow the engine's exact output format, so every feature can be explored. They are not satellite observations and not scientific evidence.</>
          : <><b>Real engine output, not validated.</b> Shown for inspection only; it did not pass the critical validation checks.</>}
        <span className="banner-reason"> Why: {p.reason}</span>
      </span>
    </div>
  );
}
