"use strict";

const state = {
  status: "OPEN",
  sort: "risk",
  cases: [],
  selected: null,
  busy: false,
};

const $ = (selector) => document.querySelector(selector);

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[char]));
}

function shortRef(value, head = 12) {
  const text = String(value ?? "");
  return text.length > head ? `${text.slice(0, head)}…` : text;
}

function rate(value) {
  if (value === null || value === undefined) return "—";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function amount(value, currency = "USD") {
  if (value === null || value === undefined) return "—";
  try {
    return new Intl.NumberFormat("en-US", {
      style: "currency", currency: currency || "USD", maximumFractionDigits: 2,
    }).format(Number(value));
  } catch {
    return `${Number(value).toFixed(2)} ${currency || ""}`.trim();
  }
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  let payload = {};
  try {
    payload = await response.json();
  } catch {
    payload = { error: `HTTP ${response.status}` };
  }
  if (!response.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

// -- metrics ---------------------------------------------------------------

function renderHistogram(histogram) {
  const entries = Object.entries(histogram || {});
  const max = Math.max(1, ...entries.map(([, count]) => count));
  $("#histogram").innerHTML = entries.map(([label, count]) => `
    <div class="bar-wrap" title="${escapeHtml(label)}: ${count}">
      <div class="bar" style="height:${Math.max(2, (count / max) * 100)}%"></div>
      <span class="bar-label">${escapeHtml(label.split("-")[0])}</span>
    </div>
  `).join("");
}

function renderAlerts(alerts) {
  const region = $("#alert-region");
  const list = $("#alert-list");
  if (!alerts || alerts.length === 0) {
    region.hidden = true;
    list.innerHTML = '<li class="muted">No active alerts.</li>';
    return;
  }
  region.hidden = false;
  region.innerHTML = alerts.map((alert) => `
    <div class="alert alert-${escapeHtml(alert.severity)}">
      <div>
        <strong>${escapeHtml(alert.severity)} · ${escapeHtml(alert.code)}</strong>
        <p>${escapeHtml(alert.message)}</p>
      </div>
    </div>
  `).join("");
  list.innerHTML = alerts.map((alert) => `
    <li class="${escapeHtml(alert.severity)}">
      <strong>${escapeHtml(alert.code)}</strong>
      <span>${escapeHtml(alert.message)}</span>
    </li>
  `).join("");
}

function renderMetrics(metrics) {
  const reviews = metrics.reviews || {};
  const labeling = metrics.labeling || {};
  const latency = metrics.latency_ms || {};
  $("#m-open").textContent = reviews.open ?? "—";
  $("#m-open-sub").textContent = `${reviews.total ?? 0} total · ${reviews.need_info ?? 0} need info`;
  $("#m-confirmed").textContent = reviews.confirmed_fraud ?? "—";
  $("#m-reviewed-sub").textContent = `${reviews.reviewed ?? 0} reviewed · ${rate(reviews.confirmed_rate)} confirmed`;
  $("#m-capture").textContent = rate(labeling.capture_rate);
  $("#m-fpr").textContent = rate(labeling.false_positive_rate);
  $("#m-fpr").className = (labeling.false_positive_rate ?? 0) > 0.02 ? "bad" : "good";

  const p99 = Number(latency.p99 ?? 0);
  $("#m-p99").textContent = `${p99.toFixed(2)} ms`;
  $("#m-p99").className = p99 > Number(latency.target ?? 50) ? "bad" : "good";
  $("#m-p99-sub").textContent = `target ${latency.target ?? 50} ms · p50 ${Number(latency.p50 ?? 0).toFixed(2)} ms`;

  $("#m-alerts").textContent = (metrics.alerts || []).length;
  $("#window-label").textContent = `${metrics.window_size ?? 0} of ${metrics.window_limit ?? 0} scores · ${metrics.observed_total ?? 0} observed`;
  $("#model-version").textContent = `model ${metrics.model_version || "—"}`;
  renderHistogram(metrics.score_histogram);
  renderAlerts(metrics.alerts || []);
}

async function loadMetrics() {
  try {
    const metrics = await api("/api/metrics");
    renderMetrics(metrics);
  } catch (error) {
    $("#health").innerHTML = `<span class="dot"></span> offline`;
    $("#health").classList.remove("chip-live");
    $("#health").title = error.message;
  }
}

// -- queue -----------------------------------------------------------------

function riskClass(score) {
  if (Number(score) >= 90) return "hot";
  if (Number(score) >= 75) return "warm";
  return "";
}

function renderQueue() {
  const queue = $("#queue");
  $("#queue-count").textContent = `${state.cases.length} shown`;
  if (state.cases.length === 0) {
    queue.innerHTML = '<li class="queue-empty">No cases in this view.</li>';
    return;
  }
  queue.innerHTML = state.cases.map((item) => `
    <li>
      <button class="queue-item ${state.selected?.case_id === item.case_id ? "selected" : ""}"
              data-case="${escapeHtml(item.case_id)}">
        <span class="queue-title">${amount(item.amount, item.currency)} · ${escapeHtml(item.txn_id)}</span>
        <span class="queue-risk ${riskClass(item.risk_score)}">${Number(item.risk_score).toFixed(1)}</span>
        <span class="queue-signal">${escapeHtml((item.top_signals || []).slice(0, 2).join(" · ") || item.action)}</span>
        <span class="queue-meta">${escapeHtml(item.status)} · ${escapeHtml(item.action)}</span>
      </button>
    </li>
  `).join("");
}

async function loadQueue() {
  try {
    const query = new URLSearchParams({
      status: state.status, sort: state.sort, limit: "100",
    });
    const payload = await api(`/api/cases?${query.toString()}`);
    state.cases = payload.cases || [];
    renderQueue();
  } catch (error) {
    $("#queue").innerHTML = `<li class="queue-empty">${escapeHtml(error.message)}</li>`;
  }
}

// -- case detail -----------------------------------------------------------

function renderSignals(item) {
  const signals = Array.isArray(item.signal_scores) && item.signal_scores.length
    ? item.signal_scores
    : (item.top_signals || []).map((label) => [label, 0]);
  if (!signals.length) return '<p class="muted">No contributing signals recorded.</p>';
  const max = Math.max(0.1, ...signals.map(([, value]) => Math.abs(Number(value) || 0)));
  return `<ul class="signal-list">${signals.map(([label, value]) => {
    const numeric = Number(value) || 0;
    return `
      <li class="signal ${numeric < 0 ? "neg" : ""}">
        <span title="${escapeHtml(label)}">${escapeHtml(label)}</span>
        <span class="signal-bar"><i style="width:${Math.max(2, Math.abs(numeric) / max * 100)}%"></i></span>
      </li>`;
  }).join("")}</ul>`;
}

function renderChannels(channels) {
  const entries = Object.entries(channels || {});
  if (!entries.length) return '<p class="muted">No behavioral sample for this case.</p>';
  return `<ul class="channel-list">${entries.map(([name, z]) => {
    const value = Number(z) || 0;
    return `
      <li class="channel">
        <span>${escapeHtml(name.replace(/_/g, " "))}</span>
        <strong>${value.toFixed(2)}σ</strong>
        <div class="zbar ${value < 0 ? "neg" : ""}"><i style="width:${Math.min(100, Math.abs(value) / 3 * 100)}%"></i></div>
      </li>`;
  }).join("")}</ul>`;
}

function renderFacts(facts) {
  return `<div class="facts">${facts.map(([label, value]) => `
    <div class="fact"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value ?? "—")}</strong></div>
  `).join("")}</div>`;
}

function caseFacts(item) {
  return [
    ["Transaction", item.txn_id],
    ["Customer ref", shortRef(item.user_ref, 14)],
    ["Created", item.created_at],
    ["Channel", item.channel],
    ["Merchant", item.merchant_category],
    ["Device", item.device_id],
    ["Recipient", item.recipient_id],
    ["Segment", item.segment],
    ["Model", item.model_version],
    ["Behavior", item.behavior_status],
    ["Ring flag", item.ring_flag ? "yes" : "no"],
    ["2FA enrolled", item.two_factor_enrolled ? "yes" : "no"],
  ];
}

function renderActions(actions) {
  if (!actions || actions.length === 0) {
    return '<p class="muted">No investigator actions yet.</p>';
  }
  return `<ul class="action-history">${actions.map((action) => `
    <li>
      <strong>${escapeHtml(action.action)} · ${escapeHtml(action.investigator)}</strong>
      <span>${escapeHtml(action.created_at)}</span>
    </li>
  `).join("")}</ul>`;
}

function renderCase(item) {
  const container = $("#detail");
  const statusClass = `badge-${String(item.status || "").replace(/[^A-Z_]/g, "")}`;
  container.innerHTML = `
    <div class="case-head">
      <div class="case-title">
        <span class="case-amount">${amount(item.amount, item.currency)}</span>
        <span class="case-sub">${escapeHtml(item.txn_id)} · risk ${Number(item.risk_score).toFixed(1)} / 100 · ${escapeHtml(item.segment)}</span>
      </div>
      <div class="case-badges">
        <span class="badge ${statusClass}">${escapeHtml(item.status)}</span>
        <span class="badge badge-risk ${Number(item.risk_score) >= 90 ? "hot" : ""}">${escapeHtml(item.action)}</span>
        <span class="badge">${escapeHtml(item.outcome)}</span>
        ${item.degraded ? '<span class="badge">DEGRADED</span>' : ""}
      </div>
    </div>
    <div class="case-body">
      <section>
        <div class="section-title">Transaction context</div>
        ${renderFacts(caseFacts(item))}
      </section>
      <div class="split">
        <section>
          <div class="section-title">Why it was flagged</div>
          ${renderSignals(item)}
        </section>
        <section>
          <div class="section-title">Behavioral biometrics</div>
          <p class="muted" style="margin-bottom:.5rem">anomaly ${Number(item.behavior_score || 0).toFixed(1)} / 100</p>
          ${renderChannels(item.behavior_channels)}
        </section>
      </div>
      <section>
        <div class="section-title">Graph context</div>
        ${renderFacts(Object.entries(item.graph_features || {}).map(([key, value]) => [key.replace(/_/g, " "), value]))}
      </section>
      <section>
        <div class="section-title">Investigator audit</div>
        ${renderActions(item.actions)}
      </section>
      <form class="review-form" id="review-form">
        <div class="row">
          <div>
            <label for="investigator">Investigator ID</label>
            <input id="investigator" name="investigator" value="analyst-1" autocomplete="username">
          </div>
          <div>
            <label for="note">Note (optional)</label>
            <input id="note" name="note" maxlength="2000" placeholder="Evidence or rationale">
          </div>
        </div>
        <div class="actions">
          <button class="btn btn-danger" type="button" data-action="confirm_fraud">Confirm fraud</button>
          <button class="btn btn-primary" type="button" data-action="clear_fp">Clear false positive</button>
          <button class="btn" type="button" data-action="request_info">Request more info</button>
          <button class="btn btn-ghost" type="button" data-action="reopen">Reopen</button>
        </div>
        <div class="status-line" id="action-status" role="status"></div>
      </form>
    </div>
  `;
}

async function selectCase(caseId) {
  try {
    const item = await api(`/api/cases/${encodeURIComponent(caseId)}`);
    state.selected = item;
    history.replaceState(null, "", `?case=${encodeURIComponent(caseId)}`);
    renderCase(item);
    renderQueue();
  } catch (error) {
    $("#detail").innerHTML = `<div class="empty-state"><p class="status-line error">${escapeHtml(error.message)}</p></div>`;
  }
}

async function submitAction(action) {
  if (!state.selected || state.busy) return;
  const investigator = $("#investigator")?.value.trim() || "";
  const note = $("#note")?.value.trim() || "";
  if (!investigator) {
    setStatus("Investigator ID is required.", true);
    return;
  }
  state.busy = true;
  document.querySelectorAll("[data-action]").forEach((button) => { button.disabled = true; });
  setStatus("Saving investigator decision…");
  try {
    const payload = await api(
      `/api/cases/${encodeURIComponent(state.selected.case_id)}/actions`,
      { method: "POST", body: JSON.stringify({ action, investigator, note }) },
    );
    state.selected = { ...payload.case, actions: payload.actions };
    renderCase(state.selected);
    setStatus(`Recorded: ${action.replace(/_/g, " ")}.`);
    await Promise.all([loadQueue(), loadMetrics()]);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    state.busy = false;
    document.querySelectorAll("[data-action]").forEach((button) => { button.disabled = false; });
  }
}

function setStatus(message, isError = false) {
  const line = $("#action-status");
  if (!line) return;
  line.textContent = message;
  line.className = `status-line ${isError ? "error" : ""}`;
}

// -- events / boot ---------------------------------------------------------

$("#queue").addEventListener("click", (event) => {
  const button = event.target.closest("[data-case]");
  if (button) selectCase(button.dataset.case);
});

$("#detail").addEventListener("click", (event) => {
  const button = event.target.closest("[data-action]");
  if (button) submitAction(button.dataset.action);
});

$("#status-filter").addEventListener("change", (event) => {
  state.status = event.target.value;
  state.selected = null;
  loadQueue();
});

$("#sort-filter").addEventListener("change", (event) => {
  state.sort = event.target.value;
  loadQueue();
});

async function loadHealth() {
  try {
    const health = await api("/api/health");
    $("#health").innerHTML = `<span class="dot"></span> live · ${health.reviews.open ?? 0} open`;
    $("#health").classList.add("chip-live");
  } catch (error) {
    $("#health").innerHTML = "<span class=\"dot\"></span> offline";
    $("#health").title = error.message;
  }
}

const deepLinkCase = new URLSearchParams(location.search).get("case");
loadHealth();
loadMetrics();
loadQueue();
if (deepLinkCase) selectCase(deepLinkCase);
setInterval(loadMetrics, 5000);
setInterval(loadQueue, 10000);
