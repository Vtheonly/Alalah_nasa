import {
  clamp,
  MONTH_COUNT,
  type CalendarCell,
  type FireCollection,
  type FireFeature,
  type MonthlyMetric,
  type Region,
} from "./data";

function stableNoise(seed: number): number {
  const value = Math.sin(seed * 127.1 + 311.7) * 43758.5453;
  return value - Math.floor(value);
}

function makeSeries(region: Region): MonthlyMetric[] {
  return Array.from({ length: MONTH_COUNT }, (_, index) => {
    const year = 2000 + Math.floor(index / 12);
    const month = (index % 12) + 1;
    const seasonalPeak = region.center[1] < 0 ? 8 : 2;
    const seasonalDistance = Math.min(
      Math.abs(month - seasonalPeak),
      12 - Math.abs(month - seasonalPeak),
    );
    const season = 0.3 + 0.7 * Math.exp(-(seasonalDistance ** 2) / 18);
    const cycle = 0.86 + 0.18 * Math.sin(index * 0.23 + stableNoise(index) * 3);
    const trend = 0.86 + ((year - 2000) / 26) * 0.16;
    const event =
      (region.id === "black-summer" && year === 2019 && month >= 11) ||
      (region.id === "chilecito" && year === 2026 && month >= 7) ||
      (region.id === "amazon" && year === 2019 && month >= 8)
        ? 1.8
        : 1;
    const value = (32 + stableNoise(index + region.iso.length) * 42) * season * cycle * trend * event;
    const sigma = year < 2012 ? 0.38 : 0.23;
    const measuredFraction = clamp(
      (year < 2012 ? 0.42 : 0.69) + 0.18 * Math.sin(index * 0.37),
      0.12,
      0.98,
    );
    return {
      month: `${year}-${String(month).padStart(2, "0")}`,
      value: Number(value.toFixed(2)),
      q05: Number((value * Math.exp(-1.645 * sigma)).toFixed(2)),
      q95: Number((value * Math.exp(1.645 * sigma)).toFixed(2)),
      rawCount: Math.round(value * (year >= 2012 ? 6.2 : 3.9)),
      anomaly: Number((event * (0.9 + stableNoise(index + 20) * 0.35)).toFixed(2)),
      measuredFraction: Number(measuredFraction.toFixed(2)),
      qualityFlag: year >= 2026 && month >= 9 ? "CLIMATOLOGY_CLOUD_FALLBACK" : "NOMINAL",
    };
  });
}

function makeCalendar(series: MonthlyMetric[]): CalendarCell[] {
  return series.map((item) => ({
    year: Number(item.month.slice(0, 4)),
    month: Number(item.month.slice(5, 7)),
    value: item.value,
    q05: item.q05,
    q95: item.q95,
    rawCount: item.rawCount,
    anomaly: item.anomaly,
    measuredFraction: item.measuredFraction,
    qualityFlag: item.qualityFlag,
  }));
}

function hexagon(longitude: number, latitude: number, width: number, height: number): number[][] {
  const points = Array.from({ length: 6 }, (_, index) => {
    const angle = (Math.PI / 3) * index;
    return [
      longitude + Math.cos(angle) * width,
      latitude + Math.sin(angle) * height,
    ];
  });
  return [...points, points[0]];
}

function makeFireCells(region: Region, month: string): FireCollection {
  const [west, south, east, north] = region.bbox;
  const columns = 25;
  const rows = 20;
  const dx = (east - west) / columns;
  const dy = (north - south) / rows;
  const selectedYear = Number(month.slice(0, 4));
  const selectedMonth = Number(month.slice(5, 7));
  const eventScale =
    (region.id === "black-summer" && selectedYear === 2019 && selectedMonth >= 11) ||
    (region.id === "chilecito" && selectedYear === 2026 && selectedMonth >= 7) ||
    (region.id === "amazon" && selectedYear === 2019 && selectedMonth >= 8)
      ? 1.5
      : 1;
  const features: FireFeature[] = [];

  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const longitude = west + (column + 0.5) * dx;
      const latitude = south + (row + 0.5) * dy;
      const heat = region.hotspots.reduce((sum, [hotLon, hotLat, strength]) => {
        const distance =
          ((longitude - hotLon) / Math.max(dx * 5, 0.08)) ** 2 +
          ((latitude - hotLat) / Math.max(dy * 5, 0.08)) ** 2;
        return sum + strength * Math.exp(-distance * 1.5);
      }, 0);
      const noise = stableNoise(row * columns + column + selectedMonth * 3);
      const frp = Math.max(0, (heat * 0.78 + noise * 0.16 - 0.13) * eventScale);
      if (frp < 0.045) continue;

      const anomaly = clamp(0.82 + heat * 1.2 + noise * 0.6, 0.65, 3.8);
      const detections = Math.round(1 + frp * 34 + noise * 4);
      const feature: FireFeature = {
        type: "Feature",
        geometry: {
          type: "Polygon",
          coordinates: [hexagon(longitude, latitude, dx * 0.52, dy * 0.54)],
        },
        properties: {
          frp_score: frp,
          anomaly_score: anomaly,
          uncertainty_score: Number((frp * 0.28 + noise * 0.06).toFixed(3)),
          value: frp,
          phi_q50_mw: Number((frp * 40).toFixed(2)),
          anomaly_ratio: Number(anomaly.toFixed(2)),
          width: Number((frp * 20).toFixed(2)),
          raw_count: detections,
          quality_flag: selectedYear >= 2026 && selectedMonth >= 9
            ? "CLIMATOLOGY_CLOUD_FALLBACK"
            : "NOMINAL",
          attribution_flag: anomaly > 1.6 ? "REAL_FIRE_CHANGE" : "WITHIN_NORMAL_VARIABILITY",
        },
      };
      features.push(feature);
    }
  }

  const maxFrp = Math.max(0.01, ...features.map((feature) => feature.properties.frp_score));
  const maxAnomaly = Math.max(1, ...features.map((feature) => feature.properties.anomaly_score));
  const maxUncertainty = Math.max(1, ...features.map((feature) => feature.properties.uncertainty_score));
  for (const feature of features) {
    feature.properties.frp_score /= maxFrp;
    feature.properties.anomaly_score /= maxAnomaly;
    feature.properties.uncertainty_score /= maxUncertainty;
  }

  return { type: "FeatureCollection", features };
}

export function makeDemoData(region: Region, month: string) {
  const series = makeSeries(region);
  return {
    series,
    calendar: makeCalendar(series),
    fireCells: makeFireCells(region, month),
  };
}

export function makeDemoMap(region: Region, month: string): FireCollection {
  return makeFireCells(region, month);
}

