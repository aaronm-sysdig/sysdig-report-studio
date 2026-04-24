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
  severityCritical: "#FF7774",
  severityHigh: "#FFA940",
  severityMedium: "#FDD835",
  severityLow: "#A8ABB1",
  fixedGreen: "#4ADE80",
  falcoBlue: "#00CBE2",
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
    fontSize: 10,
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
      fontSize: 10,
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
  textStyle: { color: CHART_COLORS.black, fontSize: 11 },
};

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
