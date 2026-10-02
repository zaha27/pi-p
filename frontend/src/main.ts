import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "./style.css";
import { fit, flowField, sparkline, stackedArea } from "./charts";
import type { Meta, Metrics, ServerMessage } from "./types";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const ui = {
  form: $<HTMLFormElement>("controls"),
  file: $<HTMLInputElement>("file"),
  fileName: $("file-name"),
  model: $<HTMLSelectElement>("model"),
  slice: $<HTMLSelectElement>("slice"),
  conf: $<HTMLInputElement>("conf"),
  confValue: $<HTMLOutputElement>("conf-value"),
  track: $<HTMLInputElement>("track"),
  realtime: $<HTMLInputElement>("realtime"),
  start: $<HTMLButtonElement>("start"),
  stop: $<HTMLButtonElement>("stop"),
  pause: $<HTMLButtonElement>("pause"),
  pausedLabel: $("paused-label"),
  stage: document.querySelector(".stage") as HTMLElement,
  message: $("message"),
  live: $<HTMLCanvasElement>("live"),
  placeholder: $("placeholder"),
  tracked: $("tracked"),
  liveNote: $("live-note"),
  timeline: $<HTMLCanvasElement>("timeline"),
  flow: $<HTMLCanvasElement>("flow"),
  kpiFrame: $("kpi-frame"),
  kpiUnique: $("kpi-unique"),
  kpiFlow: $("kpi-flow"),
  kpiMs: $("kpi-ms"),
  share: $("share"),
  classes: $<HTMLUListElement>("classes"),
  congestion: $<HTMLImageElement>("congestion"),
  congestionEmpty: $("congestion-empty"),
  saturation: $("saturation"),
  statusModel: $("status-model"),
  statusFrame: $("status-frame"),
  statusTime: $("status-time"),
  downloads: $("downloads"),
  dlVideo: $<HTMLAnchorElement>("dl-video"),
  dlCsv: $<HTMLAnchorElement>("dl-csv"),
};

let videoId: string | null = null;
let socket: WebSocket | null = null;
let meta: Meta | null = null;
let latest: Metrics | null = null;
let frame: ImageBitmap | null = null;
let congestionUrl: string | null = null;
let classKeys = "";
let view = "detections";
let scheduled = false;
let paused = false;

const color = (id: number) => meta?.colors[id] ?? "#8c887c";
const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
const clock = (s: number) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const modelLabel = (path: string) => (path.endsWith("best.pt") ? path.split(/[\\/]/).at(-3) ?? path : path);

function say(text: string, error = false): void {
  ui.message.textContent = text;
  ui.message.classList.toggle("error", error);
}

async function loadModels(): Promise<void> {
  try {
    const res = await fetch("/api/models");
    const { models } = (await res.json()) as { models: string[] };
    ui.model.replaceChildren(...models.map((m) => new Option(modelLabel(m), m)));
  } catch {
    say("Nu pot contacta serverul", true);
  }
}

function upload(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append("file", file);
    xhr.upload.onprogress = (e) => e.lengthComputable && say(`Se încarcă ${Math.round((e.loaded / e.total) * 100)}%`);
    xhr.onload = () => {
      let body: { id?: string; detail?: string } = {};
      try {
        body = JSON.parse(xhr.responseText);
      } catch {}
      if (xhr.status === 200 && body.id) resolve(body.id);
      else reject(new Error(body.detail ?? `Încărcarea a eșuat (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error("Încărcarea a eșuat"));
    xhr.open("POST", "/api/videos");
    xhr.send(form);
  });
}

function setPaused(value: boolean): void {
  paused = value;
  ui.pause.classList.toggle("paused", paused);
  ui.pause.setAttribute("aria-label", paused ? "Continuă" : "Pauză");
  ui.pausedLabel.hidden = !paused;
}

function togglePause(): void {
  if (socket?.readyState !== WebSocket.OPEN || !meta) return;
  socket.send(JSON.stringify({ action: paused ? "resume" : "pause" }));
  setPaused(!paused);
  say(paused ? "Pauză" : "Procesare în curs");
}

function setRunning(running: boolean): void {
  ui.pause.hidden = !running;
  ui.stage.classList.toggle("running", running);
  if (!running) setPaused(false);
  ui.start.disabled = running || !videoId;
  ui.stop.disabled = !running;
  for (const input of [ui.file, ui.model, ui.slice, ui.track, ui.realtime]) input.disabled = running;
}

function sendParams(): void {
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ action: "params", conf: Number(ui.conf.value), view }));
  }
}

function reset(): void {
  latest = null;
  meta = null;
  classKeys = "";
  frame?.close();
  frame = null;
  fit(ui.live);
  ui.placeholder.hidden = false;
  ui.tracked.hidden = true;
  ui.classes.replaceChildren();
  ui.share.replaceChildren();
  ui.congestion.hidden = true;
  ui.congestionEmpty.hidden = false;
  ui.downloads.hidden = true;
}

function start(): void {
  if (!videoId || socket) return;
  reset();
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/stream`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () =>
    ws.send(
      JSON.stringify({
        action: "start",
        video: videoId,
        weights: ui.model.value,
        conf: Number(ui.conf.value),
        slice: Number(ui.slice.value),
        track: ui.track.checked,
        realtime: ui.realtime.checked,
      }),
    );
  ws.onmessage = (e) => (typeof e.data === "string" ? onMessage(JSON.parse(e.data)) : onImage(e.data));
  ws.onerror = () => say("Conexiunea cu serverul s-a întrerupt", true);
  ws.onclose = () => {
    if (!meta && !ui.message.classList.contains("error")) say("Serverul a închis conexiunea înainte de pornire", true);
    socket = null;
    setRunning(false);
  };
  socket = ws;
  setRunning(true);
  say("Se pregătește modelul…");
}

async function onImage(buf: ArrayBuffer): Promise<void> {
  const blob = new Blob([new Uint8Array(buf, 1)], { type: "image/jpeg" });
  if (new Uint8Array(buf, 0, 1)[0] === 1) {
    const bitmap = await createImageBitmap(blob);
    frame?.close();
    frame = bitmap;
    schedule();
    return;
  }
  if (congestionUrl) URL.revokeObjectURL(congestionUrl);
  congestionUrl = URL.createObjectURL(blob);
  ui.congestion.src = congestionUrl;
  ui.congestion.hidden = false;
  ui.congestionEmpty.hidden = true;
}

function onMessage(msg: ServerMessage): void {
  switch (msg.type) {
    case "meta":
      meta = msg;
      ui.statusModel.textContent = `${msg.model} · ${msg.device}`;
      ui.liveNote.textContent = `sursă ${msg.width}×${msg.height} · ${Math.round(msg.fps)} fps`;
      ui.placeholder.hidden = true;
      ui.tracked.hidden = !msg.track;
      say("Procesare în curs");
      sendParams();
      break;
    case "metrics":
      latest = msg;
      schedule();
      break;
    case "done":
      say("Procesare încheiată");
      ui.dlVideo.href = msg.files["detectii.mp4"] ?? "#";
      ui.dlCsv.href = msg.files["numaratoare.csv"] ?? "#";
      ui.downloads.hidden = !Object.keys(msg.files).length;
      break;
    case "error":
      say(msg.message, true);
      break;
  }
}

function schedule(): void {
  if (scheduled) return;
  scheduled = true;
  requestAnimationFrame(draw);
}

function drawFrame(bitmap: ImageBitmap): void {
  const [ctx, w, h] = fit(ui.live);
  const s = Math.min(w / bitmap.width, h / bitmap.height);
  const dw = bitmap.width * s;
  const dh = bitmap.height * s;
  ctx.drawImage(bitmap, (w - dw) / 2, (h - dh) / 2, dw, dh);
}

function renderClasses(m: Metrics): void {
  const keys = m.classes.map((c) => c.id).join(",");
  if (keys !== classKeys) {
    classKeys = keys;
    ui.classes.replaceChildren(
      ...m.classes.map((c) => {
        const li = document.createElement("li");
        li.innerHTML = '<span class="tick"></span><span class="name"></span><canvas></canvas><span class="count"></span><span class="pct"></span>';
        (li.querySelector(".tick") as HTMLElement).style.background = color(c.id);
        li.querySelector(".name")!.textContent = meta?.names[c.id] ?? String(c.id);
        return li;
      }),
    );
    ui.share.replaceChildren(
      ...m.classes.map((c) => {
        const span = document.createElement("span");
        span.style.background = color(c.id);
        return span;
      }),
    );
  }
  m.classes.forEach((c, i) => {
    const li = ui.classes.children[i];
    li.querySelector(".count")!.textContent = String(c.count);
    li.querySelector(".pct")!.textContent = pct(c.share);
    sparkline(li.querySelector("canvas")!, c.spark, color(c.id));
    const span = ui.share.children[i] as HTMLElement;
    span.style.flexGrow = String(c.share);
    span.textContent = c.share >= 0.12 ? pct(c.share) : "";
  });
}

function draw(): void {
  scheduled = false;
  if (frame) drawFrame(frame);
  if (!latest || !meta) return;
  const m = latest;
  ui.statusFrame.textContent = `${m.frame} / ${meta.total || "?"}`;
  ui.statusTime.textContent = clock(m.time);
  ui.kpiFrame.textContent = String(m.in_frame);
  ui.kpiUnique.textContent = meta.track ? String(m.unique) : "—";
  ui.kpiFlow.textContent = meta.track ? m.mean_flow.toFixed(1) : "—";
  ui.kpiMs.textContent = m.inference_ms.toFixed(0);
  ui.tracked.textContent = `${m.in_frame} urmărite`;
  ui.saturation.textContent = pct(m.saturation);
  renderClasses(m);
  stackedArea(ui.timeline, m.timeline.series, m.classes.map((c) => color(c.id)), m.timeline.seconds);
  flowField(ui.flow, m.flow, meta.width / meta.height, color);
}

ui.file.addEventListener("change", async () => {
  const file = ui.file.files?.[0];
  if (!file) return;
  ui.fileName.textContent = file.name;
  videoId = null;
  ui.start.disabled = true;
  try {
    videoId = await upload(file);
    say("Video încărcat. Apasă Pornește.");
    ui.start.disabled = false;
  } catch (e) {
    say((e as Error).message, true);
  }
});

ui.form.addEventListener("submit", (e) => {
  e.preventDefault();
  start();
});

ui.stop.addEventListener("click", () => socket?.send(JSON.stringify({ action: "stop" })));

ui.stage.addEventListener("click", togglePause);

document.addEventListener("keydown", (e) => {
  if (e.code !== "Space" || (e.target instanceof Element && e.target.closest("input, select, button"))) return;
  e.preventDefault();
  togglePause();
});

ui.conf.addEventListener("input", () => {
  ui.confValue.value = Number(ui.conf.value).toFixed(2);
  sendParams();
});

document.querySelectorAll<HTMLButtonElement>(".segmented button").forEach((button) =>
  button.addEventListener("click", () => {
    view = button.dataset.view!;
    document.querySelectorAll(".segmented button").forEach((b) => b.classList.toggle("on", b === button));
    sendParams();
  }),
);

window.addEventListener("resize", schedule);

loadModels();
draw();
flowField(ui.flow, [], 16 / 9, color);
stackedArea(ui.timeline, [], [], 10);
