import { Clock, CloudOff, Copy, Factory, Gauge, GitMerge, Satellite, Scan, type LucideIcon } from "lucide-react";

const ICONS: Record<string, LucideIcon> = {
  "git-merge": GitMerge, scan: Scan, clock: Clock, "cloud-off": CloudOff,
  satellite: Satellite, factory: Factory, copy: Copy, gauge: Gauge,
};

export default function Icon({ name, size = 18 }: { name: string; size?: number }) {
  const I = ICONS[name] ?? Gauge;
  return <I size={size} />;
}
