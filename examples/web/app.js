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

// whitespace-preserving cap for tool payloads (short() is for one-liners)
const cap = (v, n = 4000) => {
  const s = typeof v === "string" ? v : JSON.stringify(v, null, 2);
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

function pre(cls, text) {
  const el = document.createElement("pre");
  el.className = cls;
  el.textContent = text;
  return el;
}

// one tool call: ● name, its input, its output — collapsible
function toolBlock(name, args, open) {
  const d = document.createElement("details");
  d.className = "tool";
  d.open = open;
  const s = document.createElement("summary");
  s.textContent = "● " + name;
  d.append(s);
  if (args) d.append(pre("io args", args));
  const result = pre("io result", "…");
  d.append(result);
  $("messages").append(d);
  $("messages").scrollTop = $("messages").scrollHeight;
  return { d, result };
}

function finishTool(t, result, isError) {
  t.result.textContent = result || "(no output)";
  if (isError) t.d.classList.add("err");
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
    if (onclick && it.id) {
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

// a listing entry is clickable unless it is the "… N more" tail
const files = (names) => names.map((n) => ({ id: n.startsWith("…") ? null : n, label: n }));

async function refreshAdapters() {
  const a = await api("/api/adapters");
  list("mounted", a.mounted.map((m) => ({ label: m.name, note: m.notes.join(" ") })));
  list("internal", a.internal.map((n) => ({ label: n })));
  list("external", files(a.external), (p) => openFile("external", p));
  list("workspace", files(a.workspace), (p) => openFile("workspace", p));
}

// ---- the file viewer (read-only) --------------------------------------------

async function openFile(root, path) {
  const f = await api(`/api/file?root=${root}&path=${encodeURIComponent(path)}`);
  $("viewer-name").textContent = f.path;
  $("viewer-body").textContent = f.content;   // server output, still not innerHTML
  $("viewer").hidden = false;
}

const closeViewer = () => { $("viewer").hidden = true; };

async function openSession(id) {
  current = id;
  $("messages").replaceChildren();
  for (const m of await api("/api/sessions/" + id)) {
    if (m.role === "tool") finishTool(toolBlock(m.name, m.args, false), m.result, m.error);
    else bubble(m.role, m.content);
  }
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
  const open = new Map();   // call_id -> live tool block
  for await (const frame of frames(res.body)) {
    if (frame.event === "done") return;
    const e = JSON.parse(frame.data), p = e.payload;
    if (e.kind === "text" && e.phase === "delta") {
      if (p.channel === "tool_args") continue;   // full args land on tool/start
      if (!text || text.channel !== p.channel) {
        text = { channel: p.channel, el: bubble(p.channel === "text" ? "assistant" : "dim", "") };
      }
      text.el.textContent += p.text;
      $("messages").scrollTop = $("messages").scrollHeight;
      continue;
    }
    text = null;
    if (e.kind === "tool" && e.phase === "start") {
      open.set(p.call_id, toolBlock(p.name, cap(p.arguments), true));
    } else if (e.kind === "tool" && e.phase === "end") {
      const t = open.get(p.call_id) || toolBlock(p.name, "", true);
      open.delete(p.call_id);
      finishTool(t, cap(p.result), p.is_error);
      $("messages").scrollTop = $("messages").scrollHeight;
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

$("viewer-close").onclick = closeViewer;
$("viewer").onclick = (ev) => { if (ev.target.id === "viewer") closeViewer(); };
document.onkeydown = (ev) => { if (ev.key === "Escape") closeViewer(); };

(async () => {
  await refreshAdapters();
  const rows = await refreshSessions();            // newest first
  await openSession(rows.length ? rows[0].id : await create());
})().catch(fail);
