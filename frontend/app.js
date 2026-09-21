// Talks only to the existing FastAPI backend (backend/app.py -> backend/rl_bridge.py).
// This file does NOT select players, compute transfers, or decide anything -
// it only calls the API and renders exactly what the backend returns.

const API_BASE = ""; // same-origin: backend serves this page, so relative paths work
const FINAL_GAMEWEEK = 38; // real 2025-26 season data only goes up to GW38

let sessionId = null;
let seasonLog = [];

const el = (id) => document.getElementById(id);

function setLoading(isLoading) {
  el("loading").classList.toggle("hidden", !isLoading);
  el("start-btn").disabled = isLoading;
  const nextBtn = el("next-btn");
  if (nextBtn) nextBtn.disabled = isLoading || nextBtn.dataset.seasonOver === "true";
}

async function callApi(path, options) {
  const res = await fetch(API_BASE + path, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

function playerCard(p) {
  const tag = p.is_captain ? '<span class="tag tag-c">C</span>'
            : p.is_vice_captain ? '<span class="tag tag-vc">VC</span>' : "";
  return `
    <div class="player-card">
      ${tag}
      <div class="pos pos-${p.position}">${p.position}</div>
      <div class="name">${p.name}</div>
      <div class="team">${p.team}</div>
    </div>`;
}

function renderChips(availableChips) {
  const entries = Object.entries(availableChips || {});
  el("chips-remaining").innerHTML = entries.map(([name, avail]) =>
    `<span class="chip-pill ${avail ? "available" : "used"}">${name.replace("_", " ")}</span>`
  ).join("");
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
  el("session-id-label").textContent = `session: ${sessionId}`;

  const legalBadge = el("legality-badge");
  legalBadge.textContent = data.legal ? "LEGAL SQUAD" : "ILLEGAL SQUAD";
  legalBadge.className = "badge " + (data.legal ? "badge-legal" : "badge-illegal");
  el("reward-badge").textContent = `Reward: ${data.reward.toFixed(1)}`;

  el("state-bank").textContent = data.bank.toFixed(1);
  el("state-ft").textContent = data.free_transfers;
  el("state-chip-used").textContent = data.chip_used;
  el("state-chip-requested").textContent = data.ppo_action.chip_requested;
  renderChips(data.available_chips);

  el("transfer-count").textContent = data.transfers;
  el("hit-count").textContent = data.hits;
  el("transfers-in").innerHTML = data.transfers_in.length
    ? data.transfers_in.map((n) => `<li>+ ${n}</li>`).join("")
    : "<li class='muted'>None (initial squad)</li>";
  el("transfers-out").innerHTML = data.transfers_out.length
    ? data.transfers_out.map((n) => `<li>- ${n}</li>`).join("")
    : "<li class='muted'>None (initial squad)</li>";

  el("xi-count").textContent = `(${data.starting_xi.length}/11)`;
  el("starting-xi").innerHTML = data.starting_xi.map(playerCard).join("");
  el("bench").innerHTML = data.bench.map(playerCard).join("");

  seasonLog.push({ gw: data.gameweek, reward: data.reward, chip: data.chip_used });
  el("season-log").innerHTML = seasonLog
    .map((g) => `<span class="log-entry">GW${g.gw}: <strong>${g.reward.toFixed(0)}</strong>${g.chip !== "none" ? ` (${g.chip})` : ""}</span>`)
    .join("");

  const nextBtn = el("next-btn");
  if (data.gameweek >= FINAL_GAMEWEEK) {
    nextBtn.disabled = true;
    nextBtn.dataset.seasonOver = "true";
    nextBtn.textContent = "Season Complete";
  } else {
    nextBtn.disabled = false;
    nextBtn.dataset.seasonOver = "false";
    nextBtn.textContent = "Next Gameweek →";
  }
}

async function startSeason() {
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
