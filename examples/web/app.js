// The whole client: fetch for state, a hand-parsed SSE body for the run.
// Model output is untrusted — textContent everywhere, never innerHTML.
"use strict";

const $ = (id) => document.getElementById(id);
let current = null;      // session id
let streaming = false;

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(`${res.status}: ${(await res.text()).trim()}`);
  return res.json();
}

const fail = (err) => bubble("error", String(err.message || err));

const short = (v, n = 120) => {
  const s = String(v).replace(/\s+/g, " ");
  return s.length > n ? s.slice(0, n) + "…" : s;
};

// ---- rendering --------------------------------------------------------------

function bubble(role, text) {
  const el = document.createElement("div");
  el.className = "msg " + role;
  el.textContent = text;
  $("messages").append(el);
  $("messages").scrollTop = $("messages").scrollHeight;
  return el;
}

function list(id, items, onclick) {
  const ul = $(id);
  ul.replaceChildren();
  for (const it of items) {
    const li = document.createElement("li");
    li.textContent = it.label;
    if (it.note) {
      const small = document.createElement("small");
      small.textContent = " " + it.note;
      li.append(small);
    }
    if (it.id) {
      li.className = it.id === current ? "on" : "";
      li.onclick = () => onclick(it.id).catch(fail);
    }
    ul.append(li);
  }
}

// ---- panels -----------------------------------------------------------------

async function refreshSessions() {
  const rows = await api("/api/sessions");
  list("sessions", rows.map((s) => ({ id: s.id, label: s.id, note: `${s.messages} msg` })),
       openSession);
  return rows;
}

async function refreshAdapters() {
  const a = await api("/api/adapters");
  list("mounted", a.mounted.map((m) => ({ label: m.name, note: m.notes.join(" ") })));
  list("internal", a.internal.map((n) => ({ label: n })));
  list("external", a.external.map((n) => ({ label: n })));
}

async function openSession(id) {
  current = id;
  $("messages").replaceChildren();
  for (const m of await api("/api/sessions/" + id)) bubble(m.role, m.content);
  await refreshSessions();
}

// ---- the run ----------------------------------------------------------------

async function* frames(body) {
  const reader = body.getReader(), decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buf += decoder.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const frame = { event: "message", data: "" };
      for (const line of buf.slice(0, i).split("\n")) {
        if (line.startsWith("event:")) frame.event = line.slice(6).trim();
        else if (line.startsWith("data:")) frame.data += line.slice(5).trim();
      }
      buf = buf.slice(i + 2);
      yield frame;
    }
  }
}

async function run(input) {
  const res = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session: current, input }),
  });
  if (!res.ok) throw new Error(`${res.status}: ${(await res.text()).trim()}`);

  let text = null;   // the assistant bubble currently being appended to
  for await (const frame of frames(res.body)) {
    if (frame.event === "done") return;
    const e = JSON.parse(frame.data), p = e.payload;
    if (e.kind === "text" && e.phase === "delta") {
      if (!text || text.channel !== p.channel) {
        text = { channel: p.channel, el: bubble(p.channel === "text" ? "assistant" : "dim", "") };
      }
      text.el.textContent += p.text;
      $("messages").scrollTop = $("messages").scrollHeight;
      continue;
    }
    text = null;
    if (e.kind === "tool" && e.phase === "start") {
      bubble("tool", `● ${p.name}(${short(JSON.stringify(p.arguments))})`);
    } else if (e.kind === "tool" && e.phase === "end") {
      bubble("tool", `  ⎿ ${short(p.result)}`);
    } else if (e.kind === "run" && e.phase === "end") {
      const err = p.error ? ` — ${p.error.type}: ${short(p.error.message)}` : "";
      bubble(p.error ? "error" : "status", `[${p.status}]${err}`);
    }
  }
  throw new Error("stream ended without a done frame");
}

// ---- wiring -----------------------------------------------------------------

function busy(on) {
  streaming = on;
  $("input").disabled = $("send").disabled = $("new").disabled = on;
  if (!on) $("input").focus();
}

$("composer").onsubmit = async (ev) => {
  ev.preventDefault();
  const input = $("input").value.trim();
  if (!input || streaming) return;
  $("input").value = "";
  bubble("user", input);
  busy(true);
  try {
    await run(input);
  } catch (err) {
    fail(err);
  } finally {
    busy(false);
    await Promise.all([refreshAdapters(), refreshSessions()]).catch(fail);
  }
};

const create = () => api("/api/sessions", { method: "POST" }).then((s) => s.id);

$("new").onclick = () => create().then(openSession).catch(fail);

(async () => {
  await refreshAdapters();
  const rows = await refreshSessions();            // newest first
  await openSession(rows.length ? rows[0].id : await create());
})().catch(fail);
