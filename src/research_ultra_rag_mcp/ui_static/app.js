"use strict";

const state = {
  status: null,
  sources: [],
  excludedSources: [],
  busy: false,
};

const byId = (id) => document.getElementById(id);

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined && text !== null) element.textContent = String(text);
  return element;
}

function button(label, action, value, className = "action-button") {
  const element = node("button", className, label);
  element.type = "button";
  element.dataset.action = action;
  if (value !== undefined) element.dataset.value = value;
  return element;
}

function listValue(value) {
  return String(value || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function formatNumber(value) {
  return new Intl.NumberFormat().format(Number(value || 0));
}

function formatDate(value) {
  if (!value) return "Not yet created";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.valueOf())) return String(value);
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(parsed);
}

function compactId(value) {
  const text = String(value || "");
  if (text.length <= 18) return text;
  return `${text.slice(0, 9)}…${text.slice(-6)}`;
}

function authorLine(source) {
  const authors = Array.isArray(source.authors) ? source.authors.filter(Boolean) : [];
  const parts = [];
  if (authors.length) parts.push(authors.join("; "));
  if (source.year) parts.push(String(source.year));
  return parts.join(" · ") || "Authorship and year not reviewed";
}

function locatorLabel(locator) {
  if (!locator) return "Source passage";
  if (locator.type === "pdf_page") {
    return `Page ${locator.page_label || locator.page || "?"}`;
  }
  return locator.section_title || locator.href || `EPUB section ${locator.section_index || "?"}`;
}

function sourceForDocument(documentId) {
  return state.sources.find((source) => source.document_id === documentId) || null;
}

function setConnection(kind, label) {
  const element = byId("connection-state");
  element.dataset.state = kind;
  element.lastElementChild.textContent = label;
}

function setBusy(active, message = "Working…") {
  state.busy = active;
  byId("busy-bar").hidden = !active;
  byId("busy-message").textContent = message;
  document.querySelectorAll("button[type='submit']").forEach((element) => {
    element.disabled = active;
  });
  byId("refresh-button").disabled = active;
  byId("ingest-button").disabled = active;
}

function toast(message, isError = false) {
  const item = node("div", `toast${isError ? " is-error" : ""}`, message);
  item.setAttribute("role", isError ? "alert" : "status");
  byId("toast-region").append(item);
  window.setTimeout(() => item.remove(), isError ? 7000 : 4200);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...(options.headers || {}),
    },
  });
  let payload = null;
  try {
    payload = await response.json();
  } catch (_error) {
    // A non-JSON response is represented by the HTTP status below.
  }
  if (!response.ok) {
    throw new Error(payload?.error || `Request failed with status ${response.status}`);
  }
  return payload;
}

function changeSummary(changes) {
  if (!changes) return "The source collection differs from the selected generation.";
  const parts = [];
  if (changes.added?.length) parts.push(`${changes.added.length} added`);
  if (changes.removed?.length) parts.push(`${changes.removed.length} removed`);
  if (changes.modified?.length) parts.push(`${changes.modified.length} modified`);
  if (changes.metadata_changed) parts.push("metadata changed");
  if (changes.source_exclusions_changed) parts.push("source inclusion changed");
  return parts.length
    ? `Pending changes: ${parts.join(", ")}. The existing generation remains searchable.`
    : "The source collection differs from the selected generation.";
}

function configureRetrieval(status) {
  const available = new Set(status.available_retrieval_methods || []);
  const radios = [...document.querySelectorAll("input[name='retrieval_method']")];
  radios.forEach((radio) => {
    radio.disabled = status.ready && !available.has(radio.value);
  });
  const checked = radios.find((radio) => radio.checked && !radio.disabled);
  if (!checked) {
    const preferred = radios.find(
      (radio) => radio.value === status.default_retrieval_method && !radio.disabled,
    );
    (preferred || radios.find((radio) => !radio.disabled) || radios[0]).checked = true;
  }
  byId("search-button").disabled = !status.ready || state.busy;
}

function renderStatus(status) {
  state.status = status;
  const projectPath = status.project_root || "";
  byId("project-name").textContent = projectPath.split(/[\\/]/).filter(Boolean).pop() || "Research project";
  byId("project-path").textContent = projectPath;
  byId("project-path").title = projectPath;
  byId("searchable-count").textContent = formatNumber(
    status.searchable_source_count ?? status.selected_source_count,
  );
  byId("source-detail").textContent = status.ready
    ? `${formatNumber(status.indexed_source_count)} indexed · ${formatNumber(status.excluded_source_count)} excluded`
    : `${formatNumber(status.selected_source_count)} ready to ingest · ${formatNumber(status.excluded_source_count)} excluded`;
  byId("chunk-count").textContent = status.ready ? formatNumber(status.chunk_count) : "0";
  byId("generation-label").textContent = status.generation_id ? compactId(status.generation_id) : "None";
  byId("generation-label").title = status.generation_id || "";
  byId("generation-date").textContent = formatDate(status.created_at);

  let indexState = "Not built";
  if (status.ready && status.stale) indexState = "Stale";
  else if (status.hybrid_ready) indexState = "Hybrid ready";
  else if (status.ready) indexState = "BM25 only";
  byId("index-state").textContent = indexState;
  byId("retrieval-detail").textContent = status.ready
    ? (status.available_retrieval_methods || []).join(" · ").toUpperCase()
    : "PDF + EPUB sources";

  const notice = byId("status-notice");
  const noticeAction = byId("notice-action");
  if (!status.ready) {
    notice.hidden = false;
    byId("status-notice-title").textContent = "No generation exists";
    byId("status-notice-text").textContent = status.message || "Create the first knowledge-base generation.";
    noticeAction.textContent = "Create generation";
    noticeAction.dataset.action = "ingest";
  } else if (status.stale) {
    notice.hidden = false;
    byId("status-notice-title").textContent = "The current generation is stale";
    byId("status-notice-text").textContent = changeSummary(status.changes);
    noticeAction.textContent = "Create fresh generation";
    noticeAction.dataset.action = "ingest";
  } else {
    notice.hidden = true;
  }
  configureRetrieval(status);
}

function tagList(values, className = "tag") {
  const fragment = document.createDocumentFragment();
  (values || []).forEach((value) => fragment.append(node("span", className, value)));
  return fragment;
}

function sourceCard(source) {
  const card = node("article", "source-card");
  card.dataset.searchText = [
    source.title,
    ...(source.authors || []),
    ...(source.categories || []),
    ...(source.keywords || []),
    source.source_relative_path,
  ].join(" ").toLocaleLowerCase();

  const body = node("div", "source-card-body");
  const titleRow = node("div", "source-title-row");
  titleRow.append(node("span", "format-badge", source.format || "source"));
  titleRow.append(node("span", "source-title", source.title || source.source_relative_path));
  body.append(titleRow);
  body.append(node("p", "source-byline", authorLine(source)));
  const path = node("p", "source-path", source.source_relative_path);
  path.title = source.source_path || source.source_relative_path;
  body.append(path);
  if ((source.categories || []).length || (source.keywords || []).length) {
    const tags = node("div", "source-tags");
    tags.append(tagList(source.categories, "tag"));
    tags.append(tagList(source.keywords, "tag tag-keyword"));
    body.append(tags);
  }

  const actions = node("div", "source-card-actions");
  actions.append(button("Open original", "open-source", source.source_relative_path));
  actions.append(button("Edit metadata", "edit-metadata", source.document_id));
  actions.append(button("Exclude", "exclude-source", source.document_id));
  card.append(body, actions);
  return card;
}

function excludedCard(source) {
  const card = node("article", "excluded-item");
  const body = node("div");
  body.append(node("strong", "", source.source_relative_path));
  body.append(node("p", "excluded-reason", source.reason || "No reason recorded"));
  const status = source.exists === false
    ? "File missing"
    : source.indexed_in_current_generation
      ? "Blocked from current search"
      : "Not in current index";
  body.append(node("span", "state-badge", status));
  card.append(body, button("Restore source", "restore-source", source.source_relative_path));
  return card;
}

function renderSources(payload) {
  state.sources = payload.sources || [];
  state.excludedSources = payload.excluded_sources || [];
  byId("source-tab-count").textContent = String(state.sources.length);
  byId("excluded-count").textContent = String(state.excludedSources.length);

  const list = byId("source-list");
  list.replaceChildren();
  if (!state.sources.length) {
    list.append(node("div", "no-records", payload.ready ? "No sources match this collection." : "Create a generation to inspect indexed sources."));
  } else {
    state.sources.forEach((source) => list.append(sourceCard(source)));
  }

  const excludedSection = byId("excluded-section");
  excludedSection.hidden = !state.excludedSources.length;
  const excludedList = byId("excluded-list");
  excludedList.replaceChildren();
  state.excludedSources.forEach((source) => excludedList.append(excludedCard(source)));
}

async function loadWorkspace({ announce = false } = {}) {
  setBusy(true, "Reading the current generation…");
  setConnection("loading", "Connecting");
  try {
    const [status, sources] = await Promise.all([api("/api/status"), api("/api/sources")]);
    renderStatus(status);
    renderSources(sources);
    setConnection("ready", "Local · ready");
    if (announce) toast("Workspace refreshed.");
  } catch (error) {
    setConnection("error", "Connection failed");
    toast(error.message, true);
  } finally {
    setBusy(false);
    if (state.status) configureRetrieval(state.status);
  }
}

function switchView(name) {
  document.querySelectorAll(".tab-button").forEach((tab) => {
    const active = tab.dataset.view === name;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  document.querySelectorAll(".view-panel").forEach((panel) => {
    const active = panel.dataset.panel === name;
    panel.classList.toggle("is-active", active);
    panel.hidden = !active;
  });
}

function scoreLabel(label, value) {
  if (value === null || value === undefined) return null;
  const shown = typeof value === "number" && !Number.isInteger(value) ? value.toFixed(4) : value;
  return node("span", "score-label", `${label} ${shown}`);
}

function resultCard(hit) {
  const card = node("article", "result-card");
  card.append(node("div", "result-rank", String(hit.rank).padStart(2, "0")));
  const content = node("div", "result-content");

  const header = node("div", "result-card-header");
  header.append(node("h3", "result-title", hit.title || hit.source_path));
  header.append(node("span", "locator-badge", locatorLabel(hit.locator)));
  content.append(header);
  content.append(node("div", "result-byline", `${authorLine(hit)} · ${hit.source_path}`));
  content.append(node("p", "result-citation", hit.citation || "Citation unavailable"));
  content.append(node("p", "passage-text", hit.text || ""));

  if ((hit.categories || []).length || (hit.keywords || []).length) {
    const tags = node("div", "tag-row");
    tags.append(tagList(hit.categories, "tag"));
    tags.append(tagList(hit.keywords, "tag tag-keyword"));
    content.append(tags);
  }

  const footer = node("div", "result-footer");
  const actions = node("div", "result-actions");
  actions.append(button("Copy passage", "copy-passage", hit.chunk_id));
  actions.append(button("Copy citation", "copy-citation", hit.chunk_id));
  actions.append(button("Nearby context", "show-context", hit.chunk_id));
  const source = sourceForDocument(hit.document_id);
  if (source) actions.append(button("Open original", "open-source", source.source_relative_path));
  footer.append(actions);

  const scores = node("div", "score-row");
  [
    scoreLabel("BM25 #", hit.component_ranks?.bm25),
    scoreLabel("Dense #", hit.component_ranks?.dense),
    scoreLabel("Cosine", hit.component_scores?.dense_cosine_similarity),
    scoreLabel("RRF", hit.fusion_score),
    scoreLabel("Rerank", hit.rerank_score),
  ].filter(Boolean).forEach((item) => scores.append(item));
  const chunk = node("span", "score-label", compactId(hit.chunk_id));
  chunk.title = hit.chunk_id;
  scores.append(chunk);
  footer.append(scores);
  content.append(footer);
  card.append(content);
  return card;
}

function renderResults(payload) {
  const results = byId("results");
  results.replaceChildren();
  byId("search-empty").hidden = true;
  byId("search-summary").hidden = false;
  const count = payload.result_count || 0;
  byId("result-heading").textContent = `${count} passage${count === 1 ? "" : "s"}`;
  const details = [payload.retrieval_method?.toUpperCase(), payload.reranked ? "CPU reranked" : null, compactId(payload.generation_id)].filter(Boolean);
  byId("result-meta").textContent = details.join(" · ");
  if (!count) {
    results.append(node("div", "no-records", "No passages matched. Broaden the query or remove metadata filters."));
    return;
  }
  (payload.hits || []).forEach((hit) => {
    state.hits.set(hit.chunk_id, hit);
    results.append(resultCard(hit));
  });
  if (payload.stale) toast("Results come from the current generation, which is marked stale.");
}

function clearResults() {
  state.hits = new Map();
  byId("results").replaceChildren();
  byId("search-summary").hidden = true;
  byId("search-empty").hidden = false;
}

async function search(event) {
  event.preventDefault();
  const query = byId("query").value.trim();
  if (!query) return;
  const method = document.querySelector("input[name='retrieval_method']:checked")?.value || "hybrid";
  const payload = {
    query,
    top_k: Number(byId("top-k").value),
    categories: listValue(byId("category-filter").value),
    keywords: listValue(byId("keyword-filter").value),
    document_ids: null,
    retrieval_method: method,
    rerank: byId("rerank").checked,
  };
  setBusy(true, payload.rerank ? "Searching and CPU reranking…" : "Searching evidence…");
  try {
    state.hits = new Map();
    renderResults(await api("/api/search", { method: "POST", body: JSON.stringify(payload) }));
    byId("search-summary").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    toast(error.message, true);
  } finally {
    setBusy(false);
    if (state.status) configureRetrieval(state.status);
  }
}

function openSource(path) {
  window.open(`/api/source-file?path=${encodeURIComponent(path)}`, "_blank", "noopener");
}

async function copyText(text, message) {
  try {
    await navigator.clipboard.writeText(text);
    toast(message);
  } catch (_error) {
    toast("Clipboard access was blocked by the browser.", true);
  }
}

async function showContext(chunkId) {
  setBusy(true, "Loading nearby passages…");
  try {
    const payload = await api(`/api/passages/${encodeURIComponent(chunkId)}?context_chunks=1`);
    const container = byId("context-content");
    container.replaceChildren();
    (payload.context || []).forEach((passage) => {
      const item = node("article", `context-passage${passage.chunk_id === payload.requested_chunk_id ? " is-requested" : ""}`);
      const citation = node("div", "context-citation");
      citation.append(node("span", "", passage.citation));
      citation.append(node("span", "locator-badge", locatorLabel(passage.locator)));
      item.append(citation, node("p", "", passage.text));
      container.append(item);
    });
    if (!container.children.length) container.append(node("div", "no-records", "No context was returned."));
    byId("context-title").textContent = payload.context?.[0]?.title || "Passage context";
    byId("context-dialog").showModal();
  } catch (error) {
    toast(error.message, true);
  } finally {
    setBusy(false);
    if (state.status) configureRetrieval(state.status);
  }
}

function openMetadata(documentId) {
  const source = sourceForDocument(documentId);
  if (!source) return;
  byId("metadata-source-path").value = source.source_relative_path;
  byId("metadata-title").value = source.title || "";
  byId("metadata-authors").value = (source.authors || []).join(", ");
  byId("metadata-year").value = source.year || "";
  byId("metadata-doi").value = source.doi || "";
  byId("metadata-categories").value = (source.categories || []).join(", ");
  byId("metadata-keywords").value = (source.keywords || []).join(", ");
  byId("metadata-dialog").showModal();
}

function openExclusion(documentId) {
  const source = sourceForDocument(documentId);
  if (!source) return;
  byId("exclusion-source-path").value = source.source_relative_path;
  byId("exclusion-source-name").textContent = source.source_relative_path;
  byId("exclusion-reason").value = "";
  byId("exclusion-dialog").showModal();
}

async function saveMetadata(event) {
  event.preventDefault();
  const yearText = byId("metadata-year").value.trim();
  const payload = {
    source_path: byId("metadata-source-path").value,
    metadata: {
      title: byId("metadata-title").value.trim(),
      authors: listValue(byId("metadata-authors").value),
      year: yearText ? Number(yearText) : null,
      doi: byId("metadata-doi").value.trim(),
      categories: listValue(byId("metadata-categories").value),
      keywords: listValue(byId("metadata-keywords").value),
    },
  };
  setBusy(true, "Saving reviewed metadata…");
  try {
    const result = await api("/api/source-metadata", { method: "POST", body: JSON.stringify(payload) });
    byId("metadata-dialog").close();
    toast(result.message || "Metadata saved. Create a generation to apply it.");
    await loadWorkspace();
  } catch (error) {
    toast(error.message, true);
  } finally {
    setBusy(false);
    if (state.status) configureRetrieval(state.status);
  }
}

async function excludeSource(event) {
  event.preventDefault();
  const payload = {
    source_path: byId("exclusion-source-path").value,
    included: false,
    reason: byId("exclusion-reason").value.trim(),
  };
  setBusy(true, "Excluding source from retrieval…");
  try {
    const result = await api("/api/source-inclusion", { method: "POST", body: JSON.stringify(payload) });
    byId("exclusion-dialog").close();
    clearResults();
    toast(result.message || "Source excluded.");
    await loadWorkspace();
  } catch (error) {
    toast(error.message, true);
  } finally {
    setBusy(false);
    if (state.status) configureRetrieval(state.status);
  }
}

async function restoreSource(path) {
  setBusy(true, "Restoring source…");
  try {
    const result = await api("/api/source-inclusion", {
      method: "POST",
      body: JSON.stringify({ source_path: path, included: true }),
    });
    clearResults();
    toast(result.message || "Source restored.");
    await loadWorkspace();
  } catch (error) {
    toast(error.message, true);
  } finally {
    setBusy(false);
    if (state.status) configureRetrieval(state.status);
  }
}

async function ingest(event) {
  event.preventDefault();
  const chunkSize = Number(byId("chunk-size").value);
  const chunkOverlap = Number(byId("chunk-overlap").value);
  if (chunkOverlap >= chunkSize) {
    toast("Chunk overlap must be smaller than chunk size.", true);
    return;
  }
  setBusy(true, "Building BM25 and dense indexes. This can take several minutes…");
  try {
    const result = await api("/api/ingest", {
      method: "POST",
      body: JSON.stringify({ chunk_size: chunkSize, chunk_overlap: chunkOverlap }),
    });
    byId("ingest-dialog").close();
    clearResults();
    toast(`Generation ${compactId(result.generation_id)} is ready with ${formatNumber(result.chunk_count)} passages.`);
    await loadWorkspace();
  } catch (error) {
    toast(error.message, true);
  } finally {
    setBusy(false);
    if (state.status) configureRetrieval(state.status);
  }
}

function handleAction(event) {
  const target = event.target.closest("[data-action]");
  if (!target) return;
  const { action, value } = target.dataset;
  if (action === "open-source") openSource(value);
  else if (action === "edit-metadata") openMetadata(value);
  else if (action === "exclude-source") openExclusion(value);
  else if (action === "restore-source") restoreSource(value);
  else if (action === "show-context") showContext(value);
  else if (action === "copy-passage") {
    const hit = state.hits.get(value);
    if (hit) copyText(hit.text, "Passage copied.");
  } else if (action === "copy-citation") {
    const hit = state.hits.get(value);
    if (hit) copyText(hit.citation, "Citation copied.");
  } else if (action === "ingest") byId("ingest-dialog").showModal();
}

function filterSources(event) {
  const query = event.target.value.trim().toLocaleLowerCase();
  document.querySelectorAll(".source-card").forEach((card) => {
    card.hidden = Boolean(query) && !card.dataset.searchText.includes(query);
  });
}

function initialize() {
  state.hits = new Map();
  document.querySelectorAll(".tab-button").forEach((tab) => {
    tab.addEventListener("click", () => switchView(tab.dataset.view));
  });
  document.querySelectorAll("dialog").forEach((dialog) => {
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) dialog.close();
    });
  });
  document.querySelectorAll(".dialog-close").forEach((control) => {
    control.addEventListener("click", () => control.closest("dialog").close());
  });
  byId("refresh-button").addEventListener("click", () => loadWorkspace({ announce: true }));
  byId("ingest-button").addEventListener("click", () => byId("ingest-dialog").showModal());
  byId("notice-action").addEventListener("click", handleAction);
  byId("search-form").addEventListener("submit", search);
  byId("metadata-form").addEventListener("submit", saveMetadata);
  byId("exclusion-form").addEventListener("submit", excludeSource);
  byId("ingest-form").addEventListener("submit", ingest);
  byId("source-filter").addEventListener("input", filterSources);
  byId("results").addEventListener("click", handleAction);
  byId("source-list").addEventListener("click", handleAction);
  byId("excluded-list").addEventListener("click", handleAction);
  loadWorkspace();
}

document.addEventListener("DOMContentLoaded", initialize);
