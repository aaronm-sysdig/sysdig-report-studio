/**
 * Shared ECharts chart-config primitives for SAS widgets.
 *
 * Every time-series widget imports these so visual language stays consistent.
 * Overriding these values in a widget is discouraged — if a specific widget
 * needs different behaviour, first ask whether the tenet should change.
 *
 * See spec §2 tenet: "Flowing lines, anchored observations."
 *
 * NOTE: ECharts configs render on a <canvas> outside the DOM token scope,
 * so CSS vars (var(--...)) do not resolve. We use raw hex values that mirror
 * the brand tokens in tokens.css. Keep these in sync if brand colours change.
 */

// Brand palette used inside charts (hex mirrors of tokens.css)
export const CHART_COLORS = {
  deepSee: "#01353E",
  lumin: "#BDF78B",
  white: "#FFFFFF",
  black: "#000000",
  greyBorder: "#D4D6D9",
  greyMuted: "#6E7178",
  severityCritical: "#cb87da",
  severityHigh: "#ff7875",
  severityMedium: "#ffaa40",
  severityLow: "#fdd836",
  severityNegligible: "#b5c4cc",
  fixedGreen: "#4ADE80",
  falcoBlue: "#00CBE2",
  darkRed: "#780606"
} as const;

/**
 * Line series config that enforces the "flowing lines, anchored observations" tenet.
 *
 * Use by spreading into your `series` entry:
 *   series: [{
 *     ...flowingLineSeries({ color: CHART_COLORS.deepSee }),
 *     data: myYValues,
 *   }]
 *
 * Do NOT override `smooth` to a harder value (>0.6) or set `step` — that defeats the tenet.
 */
export function flowingLineSeries(opts: {
  color: string;
  width?: number;
  symbolSize?: number;
}) {
  return {
    type: "line" as const,
    smooth: 0.4,
    lineStyle: { color: opts.color, width: opts.width ?? 2 },
    itemStyle: { color: opts.color },
    symbol: "circle",
    symbolSize: opts.symbolSize ?? 5,
    showSymbol: true,
    emphasis: {
      scale: 1.4,
      itemStyle: { borderColor: CHART_COLORS.white, borderWidth: 2 },
    },
    connectNulls: false, // missing snapshots render as gaps (honesty)
  };
}

/**
 * Standard y-axis config for count-style widgets.
 * Dashed grid lines at each tick; no axis line; labels in brand grey.
 */
export const STANDARD_Y_AXIS = {
  type: "value" as const,
  minInterval: 1,
  axisLabel: {
    fontSize: 12,
    color: CHART_COLORS.greyMuted,
    formatter: (v: number) =>
      v >= 1000 ? `${(v / 1000).toFixed(1)}k` : String(v),
  },
  splitLine: {
    lineStyle: { color: CHART_COLORS.greyBorder, type: "dashed" as const },
  },
  axisLine: { show: false },
  axisTick: { show: false },
};

/**
 * Standard x-axis config for category (date) time-series.
 * Honours the axis-labels toggle — pass `showLabels` true to reveal date labels.
 * Cadence-smart label interval: weekly for 90d, every 3 days for 30d, daily otherwise.
 */
export function standardXAxis(dates: string[], showLabels: boolean) {
  const n = dates.length;
  let labelInterval = 0;
  if (showLabels) {
    if (n > 30) labelInterval = 6;
    else if (n > 7) labelInterval = 2;
    else labelInterval = 0;
  }
  return {
    type: "category" as const,
    data: dates,
    axisLabel: {
      show: showLabels,
      interval: labelInterval,
      rotate: n > 7 ? 45 : 0,
      fontSize: 12,
      color: CHART_COLORS.greyMuted,
    },
    axisLine: { lineStyle: { color: CHART_COLORS.greyBorder } },
    axisTick: { show: false },
  };
}

/**
 * Standard tooltip style for all widgets — white bg, subtle border, small text.
 */
export const STANDARD_TOOLTIP_STYLE = {
  backgroundColor: CHART_COLORS.white,
  borderColor: CHART_COLORS.greyBorder,
  textStyle: { color: CHART_COLORS.black, fontSize: 12 },
};

/**
 * Severity ordering for tooltip rows — Critical first, Low last.
 * Used to sort tooltip entries so the most severe findings appear at the top.
 */
export const SEVERITY_ORDER: Record<string, number> = {
  Critical: 0,
  High: 1,
  Medium: 2,
  Low: 3,
  Negligible: 4,
};

/**
 * Build a tooltip formatter for severity-stacked charts.
 *
 * Automatically sorts rows by severity (Critical → Low) and excludes the
 * "Total" line series from the detail rows (shows it as a summary footer).
 *
 * Usage:
 *   tooltip: {
 *     trigger: "axis",
 *     ...STANDARD_TOOLTIP_STYLE,
 *     formatter: severityTooltipFormatter(),
 *   },
 */
export function severityTooltipFormatter() {
  return (params: unknown[]) => {
    const arr = params as Array<{
      axisValue: string;
      seriesName: string;
      value: number;
      color: string;
    }>;
    if (!arr.length) return "";
    const date = arr[0].axisValue;
    const rows = arr
      .filter((p) => p.seriesName !== "Total")
      .sort((a, b) => (SEVERITY_ORDER[a.seriesName] ?? 99) - (SEVERITY_ORDER[b.seriesName] ?? 99))
      .map(
        (p) =>
          `<div style="display:flex;justify-content:space-between;gap:12px">` +
          `<span style="color:${p.color}">&#9632;</span>` +
          `<span style="color:${CHART_COLORS.greyMuted};flex:1;margin-left:4px">${p.seriesName}:</span>` +
          `<b>${(p.value ?? 0).toLocaleString("en-GB")}</b></div>`,
      );
    const totalEntry = arr.find((p) => p.seriesName === "Total");
    const total = totalEntry ? totalEntry.value : 0;
    return `<div style="font-size:11px;min-width:160px">
      <div style="color:${CHART_COLORS.greyMuted};margin-bottom:4px">${date}</div>
      ${rows.join("")}
      <div style="border-top:1px solid ${CHART_COLORS.greyBorder};margin-top:4px;padding-top:4px;display:flex;justify-content:space-between">
        <span style="color:${CHART_COLORS.greyMuted}">Total:</span>
        <b>${total.toLocaleString("en-GB")}</b>
      </div>
    </div>`;
  };
}

/**
 * Standard grid (padding around the chart plot).
 * Pass `extraBottom` (e.g. 32) when axis labels are on to reserve space for rotated labels.
 */
export function standardGrid(extraBottom: number = 0) {
  return {
    top: 12,
    right: 16,
    bottom: 20 + extraBottom,
    left: 48,
    containLabel: false,
  };
}
