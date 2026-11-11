// Draw dataset-analysis charts (bar, line, pie) with Chart.js from the validated JSON spec that the server
// embeds next to each result table (see ChartSpec in app/models.py). Labels are drawn on a canvas as text,
// never inserted as HTML, and nothing in the spec is executed. If anything fails (Chart.js not loaded, bad
// spec), the chart area shows a short note and the answer and the table stay as they are.
(() => {
  const TYPES = new Set(["bar", "line", "pie"]);
  const MAX_POINTS = 100;
  const PALETTE = ["#2563eb", "#16a34a", "#d97706", "#dc2626", "#7c3aed", "#0891b2",
                   "#db2777", "#65a30d", "#ea580c", "#4f46e5", "#0d9488", "#9333ea"];

  function readSpec(figure) {
    const node = figure.querySelector("script.chart-spec");
    const spec = JSON.parse(node ? node.textContent : "null");
    const valid = spec && TYPES.has(spec.type) && Array.isArray(spec.labels) && Array.isArray(spec.values)
      && spec.values.length > 0 && spec.values.length <= MAX_POINTS && spec.labels.length === spec.values.length
      && spec.values.every((v) => typeof v === "number" && Number.isFinite(v));
    if (!valid) throw new Error("invalid chart spec");
    const percentages = Array.isArray(spec.percentages) && spec.percentages.length === spec.values.length
      ? spec.percentages : null;
    return {
      type: spec.type,
      labels: spec.labels.map((label) => String(label)),
      values: spec.values.slice(),
      labelName: String(spec.label_name || ""),
      seriesName: String(spec.series_name || ""),
      percentages,
    };
  }

  function config(spec) {
    const css = getComputedStyle(document.documentElement);
    const muted = css.getPropertyValue("--muted").trim() || "#6b7280";
    const grid = css.getPropertyValue("--border").trim() || "#e5e7eb";
    const options = { responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { display: false } } };
    if (spec.type === "pie") {
      options.plugins.legend = { display: true, position: "bottom", labels: { color: muted } };
      options.plugins.tooltip = {
        callbacks: {
          label: (item) => {
            const share = spec.percentages ? ` (${spec.percentages[item.dataIndex]}%)` : "";
            return `${spec.labels[item.dataIndex]}: ${spec.values[item.dataIndex]}${share}`;
          },
        },
      };
      return {
        type: "pie",
        data: { labels: spec.labels, datasets: [{ label: spec.seriesName, data: spec.values,
                backgroundColor: spec.values.map((_, i) => PALETTE[i % PALETTE.length]) }] },
        options,
      };
    }
    const axis = (title) => ({ title: { display: Boolean(title), text: title, color: muted }, ticks: { color: muted },
                               grid: { color: grid } });
    options.scales = { x: axis(spec.labelName), y: { ...axis(spec.seriesName), beginAtZero: spec.type === "bar" } };
    return {
      type: spec.type,
      data: { labels: spec.labels, datasets: [{ label: spec.seriesName, data: spec.values, borderColor: PALETTE[0],
              backgroundColor: PALETTE[0], fill: false, tension: 0.2 }] },
      options,
    };
  }

  function showFallback(figure) {
    const box = figure.querySelector(".chart-canvas");
    if (box) box.hidden = true;
    const note = figure.querySelector(".chart-fallback");
    if (note) note.hidden = false;
    figure.dataset.chartState = "failed";
  }

  function render(figure) {
    if (figure.dataset.chartState) return; // already drawn (or failed): never a second instance
    const canvas = figure.querySelector("canvas");
    try {
      const Chart = window.Chart;
      if (typeof Chart !== "function" || !canvas) throw new Error("Chart.js is not available");
      const spec = readSpec(figure);
      const existing = typeof Chart.getChart === "function" ? Chart.getChart(canvas) : undefined;
      if (existing) existing.destroy();
      new Chart(canvas, config(spec));
      figure.dataset.chartState = "rendered";
    } catch (error) {
      console.warn("A chart could not be displayed; its table is still shown.");
      showFallback(figure);
    }
  }

  function renderAll(root) {
    if (root && typeof root.querySelectorAll === "function") root.querySelectorAll("figure.analysis-chart").forEach(render);
  }

  function destroyIn(root) {
    const Chart = window.Chart;
    if (!root || typeof root.querySelectorAll !== "function" || !Chart || typeof Chart.getChart !== "function") return;
    root.querySelectorAll("figure.analysis-chart canvas").forEach((canvas) => {
      const chart = Chart.getChart(canvas);
      if (chart) chart.destroy();
    });
  }

  document.addEventListener("DOMContentLoaded", () => renderAll(document));
  document.addEventListener("htmx:afterSwap", (event) => renderAll(event.target)); // new messages from HTMX
  document.addEventListener("htmx:beforeCleanupElement", (event) => destroyIn(event.target)); // removed content
})();
