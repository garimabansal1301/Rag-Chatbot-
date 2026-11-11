// Evals dashboard (templates/evals.html): tabs, dataset search / filter / selection, the run dialog and the
// result-detail dialog. Page content is rendered by the server; this script only reads it and calls the JSON API.
// Nothing here starts an evaluation on its own: POST /api/evals/run is sent only from the dialog's Start button,
// after the user has ticked the confirmations, and the server checks the same confirmations again.
(() => {
  const root = document.getElementById("evals");
  if (!root) return;
  const $ = (selector, scope = document) => scope.querySelector(selector);
  const $$ = (selector, scope = document) => Array.from(scope.querySelectorAll(selector));
  const selected = new Set();

  // ------------------------------------------------------------------ tabs
  const tabs = $$('[role="tab"]', root);
  function showTab(name) {
    tabs.forEach((tab) => {
      const active = tab.id === `tab-${name}`;
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
      const panel = document.getElementById(tab.getAttribute("aria-controls"));
      if (panel) panel.hidden = !active;
    });
  }
  window.evalsShowTab = showTab;
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => showTab(tab.id.replace("tab-", "")));
    tab.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
      const next = tabs[(index + (event.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
      next.focus();
      showTab(next.id.replace("tab-", ""));
    });
  });

  // ------------------------------------------------------------------ dataset: search, filter, selection
  const rows = $$("#dataset-table tbody tr");
  const search = $("#dataset-search");
  const category = $("#dataset-category");
  const selectAll = $("#dataset-select-all");

  function visibleRows() {
    return rows.filter((row) => !row.hidden);
  }

  function updateSelection() {
    const count = selected.size;
    $("#dataset-selected").textContent = `${count} selected`;
    $("#dataset-clear").disabled = count === 0;
    $("#run-selected").disabled = count === 0;
    $$("[data-selected-count]").forEach((node) => { node.textContent = String(count); });
    const shown = visibleRows();
    const shownSelected = shown.filter((row) => selected.has(row.dataset.id)).length;
    selectAll.checked = shown.length > 0 && shownSelected === shown.length;
    selectAll.indeterminate = shownSelected > 0 && shownSelected < shown.length;
    const scopeSelected = $('#run-form input[name="scope"][value="selected"]');
    if (scopeSelected) {
      scopeSelected.disabled = count === 0;
      if (count === 0 && scopeSelected.checked) $('#run-form input[name="scope"][value="all"]').checked = true;
    }
    const hint = $("[data-selected-hint]");
    if (hint) hint.hidden = count > 0;
  }

  function applyFilters() {
    const needle = search.value.trim().toLowerCase();
    const wanted = category.value;
    rows.forEach((row) => {
      row.hidden = (wanted && row.dataset.category !== wanted) || (needle && !row.dataset.search.includes(needle));
    });
    $("#dataset-empty").hidden = visibleRows().length > 0;
    updateSelection();
  }

  search.addEventListener("input", applyFilters);
  category.addEventListener("change", applyFilters);
  rows.forEach((row) => {
    const box = $(".dataset-check", row);
    box.addEventListener("change", () => {
      if (box.checked) selected.add(row.dataset.id); else selected.delete(row.dataset.id);
      row.classList.toggle("selected", box.checked);
      updateSelection();
    });
  });
  selectAll.addEventListener("change", () => {
    visibleRows().forEach((row) => {
      const box = $(".dataset-check", row);
      box.checked = selectAll.checked;
      row.classList.toggle("selected", box.checked);
      if (box.checked) selected.add(row.dataset.id); else selected.delete(row.dataset.id);
    });
    updateSelection();
  });
  $("#dataset-clear").addEventListener("click", () => {
    selected.clear();
    rows.forEach((row) => { $(".dataset-check", row).checked = false; row.classList.remove("selected"); });
    updateSelection();
  });

  // ------------------------------------------------------------------ results: status filter (re-rendered by HTMX)
  document.addEventListener("change", (event) => {
    const filter = event.target.closest && event.target.closest("[data-results-filter]");
    if (!filter) return;
    const container = document.getElementById("eval-results");
    let shown = 0;
    $$(".results-table tbody tr", container).forEach((row) => {
      row.hidden = Boolean(filter.value) && row.dataset.status !== filter.value;
      if (!row.hidden) shown += 1;
    });
    const empty = $("[data-results-empty]", container);
    if (empty) empty.hidden = shown > 0;
  });

  // ------------------------------------------------------------------ dialogs
  const runDialog = $("#run-dialog");
  const detailDialog = $("#detail-dialog");
  document.addEventListener("click", (event) => {
    const closer = event.target.closest && event.target.closest("[data-close-dialog]");
    if (closer) closer.closest("dialog").close();
  });
  document.addEventListener("htmx:afterSwap", (event) => {
    if (event.detail.target && event.detail.target.id === "eval-detail" && !detailDialog.open) detailDialog.showModal();
  });
  document.addEventListener("htmx:responseError", (event) => {
    if (event.detail.target && event.detail.target.id === "eval-detail") {
      event.detail.target.textContent = "This result could not be loaded.";
      if (!detailDialog.open) detailDialog.showModal();
    }
  });

  // ------------------------------------------------------------------ run dialog
  const form = $("#run-form");
  const startButton = $("#run-start");
  const estimateBox = $("#run-estimate");
  const errorBox = $("#run-error");
  let preview = null;
  let requestId = null;
  let previewSeq = 0;

  function newRequestId() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") return window.crypto.randomUUID();
    return `req-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`;
  }

  function choice(name) {
    const input = $(`#run-form input[name="${name}"]:checked`);
    return input ? input.value : null;
  }

  function payload() {
    const scope = choice("scope");
    return { scope, mode: choice("mode"), example_ids: scope === "selected" ? Array.from(selected).sort() : [] };
  }

  function showError(message) {
    errorBox.textContent = message || "";
    errorBox.hidden = !message;
  }

  function detailText(body, fallback) {
    if (body && typeof body.detail === "string") return body.detail;
    if (body && Array.isArray(body.detail) && body.detail.length) return "The request was not valid. Check your selection.";
    return fallback;
  }

  function refreshControls() {
    const langsmith = choice("mode") === "langsmith";
    $("[data-langsmith-only]").hidden = !langsmith;
    if (!langsmith) $('input[name="confirm_langsmith_upload"]').checked = false;
    $$("[data-mode-note]").forEach((li) => li.classList.toggle("active", li.dataset.modeNote === choice("mode")));
    const confirmed = $('input[name="confirm_gemini_usage"]').checked
      && (!langsmith || $('input[name="confirm_langsmith_upload"]').checked);
    startButton.disabled = !(preview && confirmed);
  }

  async function loadPreview() {
    const seq = ++previewSeq;
    preview = null;
    $$("[data-max-calls]").forEach((node) => { node.textContent = "…"; });
    estimateBox.textContent = "Calculating the expected usage…";
    refreshControls();
    try {
      const response = await fetch("/api/evals/preview", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload()),
      });
      const body = await response.json().catch(() => null);
      if (seq !== previewSeq) return;
      if (!response.ok) {
        estimateBox.textContent = detailText(body, "The estimate could not be calculated.");
        return;
      }
      preview = body;
      // Any change to the selection needs a fresh confirmation of the new estimate.
      $('input[name="confirm_gemini_usage"]').checked = false;
      $('input[name="confirm_langsmith_upload"]').checked = false;
      $$("[data-max-calls]").forEach((node) => { node.textContent = String(body.max_gemini_calls); });
      estimateBox.innerHTML = "";
      const strong = document.createElement("strong");
      strong.textContent = `${body.example_count} example${body.example_count === 1 ? "" : "s"} · at most ${body.max_gemini_calls} Gemini calls`;
      const detail = document.createElement("span");
      detail.className = "muted small";
      detail.textContent = `${body.answer_calls} answer call${body.answer_calls === 1 ? "" : "s"} + up to ${body.judge_calls} judge calls`
        + (body.model ? ` (model ${body.model})` : "") + ". Embeddings are computed locally.";
      estimateBox.append(strong, document.createElement("br"), detail);
    } catch (error) {
      if (seq === previewSeq) estimateBox.textContent = "The estimate could not be calculated. Check your connection.";
    } finally {
      if (seq === previewSeq) refreshControls();
    }
  }

  function openRunDialog(scope) {
    form.reset();
    if (scope === "selected" && selected.size > 0) $('#run-form input[name="scope"][value="selected"]').checked = true;
    requestId = newRequestId(); // one id per dialog opening: repeated clicks cannot start a second run
    showError("");
    updateSelection();
    runDialog.showModal();
    loadPreview();
  }

  $("#open-run-dialog").addEventListener("click", () => openRunDialog("all"));
  $("#run-selected").addEventListener("click", () => openRunDialog("selected"));
  $$('#run-form input[name="scope"], #run-form input[name="mode"]').forEach((input) => {
    input.addEventListener("change", loadPreview);
  });
  $$('#run-form input[type="checkbox"]').forEach((input) => input.addEventListener("change", refreshControls));

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!preview || startButton.disabled) return;
    startButton.disabled = true;
    startButton.textContent = "Starting…";
    showError("");
    const body = {
      ...payload(),
      request_id: requestId,
      confirm_gemini_usage: $('input[name="confirm_gemini_usage"]').checked,
      confirm_langsmith_upload: $('input[name="confirm_langsmith_upload"]').checked,
      acknowledged_max_calls: preview.max_gemini_calls,
    };
    try {
      const response = await fetch("/api/evals/run", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
      });
      const result = await response.json().catch(() => null);
      if (response.ok && result && result.job_id) {
        runDialog.close();
        window.htmx.ajax("GET", `/ui/evals/status?job=${encodeURIComponent(result.job_id)}`, {
          target: "#eval-status", swap: "outerHTML",
        });
        return;
      }
      showError(detailText(result, "The evaluation could not be started."));
      if (response.status === 409) requestId = newRequestId();
    } catch (error) {
      showError("The evaluation could not be started. Check your connection and try again.");
    } finally {
      startButton.textContent = "Start evaluation";
      refreshControls();
    }
  });

  updateSelection();
})();
