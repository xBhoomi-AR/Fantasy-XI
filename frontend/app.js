// Talks only to the existing FastAPI backend (backend/app.py -> backend/rl_bridge.py).
// This file does NOT select players, compute transfers, or decide anything -
// it only calls the API and renders exactly what the backend returns.

const API_BASE = ""; // same-origin: backend serves this page, so relative paths work
const FINAL_GAMEWEEK = 38; // real 2025-26 season data only goes up to GW38

let sessionId = null;
let seasonLog = [];

const el = (id) => document.getElementById(id);

// Guards against double-clicks causing two in-flight requests at once -
// checked (not just button.disabled) because a very fast double-click can
// fire before the browser repaints the disabled state.
let requestInFlight = false;

function setLoading(isLoading) {
  requestInFlight = isLoading;
  el("loading").classList.toggle("hidden", !isLoading);

  const startBtn = el("start-btn");
  startBtn.disabled = isLoading;

  const nextBtn = el("next-btn");
  if (!nextBtn) return;
  const seasonOver = nextBtn.dataset.seasonOver === "true";
  nextBtn.disabled = isLoading || seasonOver;
  if (isLoading) {
    nextBtn.textContent = "Loading…";
  } else if (!seasonOver) {
    nextBtn.textContent = "Next Gameweek →";
  }
}

async function callApi(path, options) {
  const res = await fetch(API_BASE + path, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

const POSITION_ORDER = { GK: 0, DEF: 1, MID: 2, FWD: 3 };
const sortByPosition = (players) =>
  [...players].sort((a, b) => (POSITION_ORDER[a.position] ?? 9) - (POSITION_ORDER[b.position] ?? 9));

function playerCard(p, isNew) {
  const tag = p.is_captain ? '<span class="tag tag-c">C</span>'
            : p.is_vice_captain ? '<span class="tag tag-vc">VC</span>' : "";
  return `
    <div class="player-card${isNew ? " is-new" : ""}">
      ${tag}
      <div class="pos pos-${p.position}">${p.position}</div>
      <div class="name">${p.name}</div>
      <div class="team">${p.team}</div>
      ${isNew ? '<div class="new-flag">NEW</div>' : ""}
    </div>`;
}

function fullSquadRow(p, isNew) {
  const role = p.is_captain ? "Captain"
             : p.is_vice_captain ? "Vice-Captain"
             : p.is_starting ? "Starting XI" : "Bench";
  const roleClass = p.is_starting ? "role-xi" : "role-bench";
  return `
    <tr>
      <td><span class="pos pos-${p.position}">${p.position}</span></td>
      <td>${p.name}${isNew ? ' <span class="new-flag inline">NEW</span>' : ""}</td>
      <td class="muted">${p.team}</td>
      <td class="${roleClass}">${role}</td>
    </tr>`;
}

// "Available" (never used), "used" (already played this season), or
// "active" (the chip actually used THIS gameweek) - three distinct states,
// all read directly from the backend response, nothing computed here.
function renderChips(availableChips, chipUsedThisGw) {
  const entries = Object.entries(availableChips || {});
  el("chips-remaining").innerHTML = entries.map(([name, avail]) => {
    const isActive = name === chipUsedThisGw;
    const cls = isActive ? "active" : avail ? "available" : "used";
    const label = name.replace("_", " ");
    return `<span class="chip-pill ${cls}">${label}${isActive ? " ★" : ""}</span>`;
  }).join("");
}

function renderGameweek(data) {
  el("landing").classList.add("hidden");
  el("season").classList.remove("hidden");
  el("infeasible-banner").classList.add("hidden");

  if (data.legal === false) {
    // MILP found no legal squad under this action - a real, handled outcome,
    // not a crash. Show it plainly and stop the season here.
    el("infeasible-banner").classList.remove("hidden");
    el("infeasible-banner").textContent =
      `GW${data.gameweek ?? "?"}: ${data.message || "No legal squad could be found for this gameweek."} ` +
      `The sequential season ends here.`;
    el("gw-number").textContent = `Gameweek ${data.gameweek ?? "?"}`;
    const nextBtn = el("next-btn");
    nextBtn.disabled = true;
    nextBtn.dataset.seasonOver = "true";
    nextBtn.textContent = "Season Complete";
    return;
  }

  el("gw-number").textContent = `Gameweek ${data.gameweek}`;
  el("gw-subtitle").textContent = data.transfers_in.length === 0 && data.transfers_out.length === 0
    ? "Initial squad for the season"
    : `Built from Gameweek ${data.gameweek - 1}'s squad + this gameweek's transfers`;
  el("session-id-label").textContent = `session: ${sessionId}`;

  const legalBadge = el("legality-badge");
  legalBadge.textContent = data.legal ? "LEGAL SQUAD" : "ILLEGAL SQUAD";
  legalBadge.className = "badge " + (data.legal ? "badge-legal" : "badge-illegal");
  el("reward-badge").textContent = `Reward: ${data.reward.toFixed(1)}`;

  el("state-bank").textContent = data.bank.toFixed(1);
  el("state-ft").textContent = data.free_transfers;
  el("state-chip-used").textContent = data.chip_used;
  el("state-chip-requested").textContent = data.ppo_action.chip_requested;
  renderChips(data.available_chips, data.chip_used);

  el("transfer-count").textContent = data.transfers;
  el("hit-count").textContent = data.hits;
  el("transfers-in").innerHTML = data.transfers_in.length
    ? data.transfers_in.map((n) => `<li>+ ${n}</li>`).join("")
    : "<li class='muted'>None (initial squad)</li>";
  el("transfers-out").innerHTML = data.transfers_out.length
    ? data.transfers_out.map((n) => `<li>- ${n}</li>`).join("")
    : "<li class='muted'>None (initial squad)</li>";

  const newNames = new Set(data.transfers_in);
  const sortedSquad = sortByPosition(data.squad);

  el("full-squad").innerHTML = sortedSquad.map((p) => fullSquadRow(p, newNames.has(p.name))).join("");
  el("xi-count").textContent = `(${data.starting_xi.length}/11)`;
  el("starting-xi").innerHTML = sortByPosition(data.starting_xi).map((p) => playerCard(p, newNames.has(p.name))).join("");
  el("bench").innerHTML = sortByPosition(data.bench).map((p) => playerCard(p, newNames.has(p.name))).join("");

  seasonLog.push({ gw: data.gameweek, reward: data.reward, chip: data.chip_used });
  el("season-log").innerHTML = seasonLog
    .map((g) => `<span class="log-entry">GW${g.gw}: <strong>${g.reward.toFixed(0)}</strong>${g.chip !== "none" ? ` (${g.chip})` : ""}</span>`)
    .join("");

  const nextBtn = el("next-btn");
  const completeBanner = el("complete-banner");
  if (data.gameweek >= FINAL_GAMEWEEK) {
    nextBtn.disabled = true;
    nextBtn.dataset.seasonOver = "true";
    nextBtn.textContent = "Season Complete";
    const total = seasonLog.reduce((sum, g) => sum + g.reward, 0);
    const chipsPlayed = seasonLog.filter((g) => g.chip !== "none").map((g) => `GW${g.gw}: ${g.chip}`);
    completeBanner.classList.remove("hidden");
    completeBanner.innerHTML =
      `<strong>Season Complete</strong> &mdash; ${seasonLog.length} gameweeks played, ` +
      `total reward ${total.toFixed(1)}. ` +
      (chipsPlayed.length ? `Chips played: ${chipsPlayed.join(", ")}.` : "No chips played.");
  } else {
    nextBtn.disabled = false;
    nextBtn.dataset.seasonOver = "false";
    nextBtn.textContent = "Next Gameweek →";
    completeBanner.classList.add("hidden");
  }
}

async function startSeason() {
  if (requestInFlight) return;
  el("start-error").classList.add("hidden");
  setLoading(true);
  try {
    const data = await callApi("/season/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    sessionId = data.session_id;
    seasonLog = [];
    renderGameweek(data);
  } catch (err) {
    el("start-error").textContent = `Could not start season: ${err.message}`;
    el("start-error").classList.remove("hidden");
  } finally {
    setLoading(false);
  }
}

async function nextGameweek() {
  if (requestInFlight) return;
  el("next-error").classList.add("hidden");
  setLoading(true);
  try {
    const data = await callApi(`/season/${sessionId}/next`, { method: "POST" });
    renderGameweek(data);
  } catch (err) {
    el("next-error").textContent = `Could not fetch next gameweek: ${err.message}`;
    el("next-error").classList.remove("hidden");
  } finally {
    setLoading(false);
  }
}

el("start-btn").addEventListener("click", startSeason);
el("next-btn").addEventListener("click", nextGameweek);
