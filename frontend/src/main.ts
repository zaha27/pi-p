import "@fontsource-variable/inter";
import "@fontsource/ibm-plex-mono/400.css";
import "./style.css";
import { fetchModels, modelLabel, upload } from "./api";
import { boot, showLogin } from "./auth";
import { initDashboard, resetDashboard, schedule, showDownloads, showImage, showMeta, showMetrics } from "./dashboard";
import { $ } from "./dom";
import { initLibrary } from "./library";
import type { ServerMessage } from "./types";

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
};

let source: { video: string } | { library: string } | null = null;
let socket: WebSocket | null = null;
let started = false;
let paused = false;
let view = "detections";

function say(text: string, error = false): void {
  ui.message.textContent = text;
  ui.message.classList.toggle("error", error);
}

async function loadModels(): Promise<void> {
  try {
    const models = await fetchModels();
    if (models) ui.model.replaceChildren(...models.map((m) => new Option(modelLabel(m), m)));
  } catch {
    say("Nu pot contacta serverul", true);
  }
}

function setPaused(value: boolean): void {
  paused = value;
  ui.pause.classList.toggle("paused", paused);
  ui.pause.setAttribute("aria-label", paused ? "Continuă" : "Pauză");
  ui.pausedLabel.hidden = !paused;
}

function togglePause(): void {
  if (socket?.readyState !== WebSocket.OPEN || !started) return;
  socket.send(JSON.stringify({ action: paused ? "resume" : "pause" }));
  setPaused(!paused);
  say(paused ? "Pauză" : "Procesare în curs");
}

function setRunning(running: boolean): void {
  ui.pause.hidden = !running;
  ui.stage.classList.toggle("running", running);
  if (!running) setPaused(false);
  ui.start.disabled = running || !source;
  ui.stop.disabled = !running;
  for (const input of [ui.file, ui.model, ui.slice, ui.track, ui.realtime]) input.disabled = running;
}

function sendParams(): void {
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ action: "params", conf: Number(ui.conf.value), view }));
  }
}

function onMessage(msg: ServerMessage): void {
  switch (msg.type) {
    case "meta":
      started = true;
      showMeta(msg);
      say("Procesare în curs");
      sendParams();
      break;
    case "metrics":
      showMetrics(msg);
      break;
    case "done":
      say("Procesare încheiată");
      showDownloads(msg.files);
      break;
    case "error":
      say(msg.message, true);
      break;
  }
}

function start(): void {
  if (!source || socket) return;
  resetDashboard();
  started = false;
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/stream`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () =>
    ws.send(
      JSON.stringify({
        action: "start",
        ...source,
        weights: ui.model.value,
        conf: Number(ui.conf.value),
        slice: Number(ui.slice.value),
        track: ui.track.checked,
        realtime: ui.realtime.checked,
      }),
    );
  ws.onmessage = (e) => (typeof e.data === "string" ? onMessage(JSON.parse(e.data)) : showImage(e.data));
  ws.onerror = () => say("Conexiunea cu serverul s-a întrerupt", true);
  ws.onclose = (e) => {
    if (e.code === 4401) showLogin("Sesiunea a expirat");
    if (!started && !ui.message.classList.contains("error")) say("Serverul a închis conexiunea înainte de pornire", true);
    socket = null;
    setRunning(false);
  };
  socket = ws;
  setRunning(true);
  say("Se pregătește modelul…");
}

ui.file.addEventListener("change", async () => {
  const file = ui.file.files?.[0];
  if (!file) return;
  ui.fileName.textContent = file.name;
  source = null;
  ui.start.disabled = true;
  try {
    source = { video: await upload(file, (p) => say(`Se încarcă ${p}%`)) };
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

initLibrary((name) => {
  source = { library: name };
  ui.file.value = "";
  ui.fileName.textContent = name;
  ui.start.disabled = !!socket;
  say("Video din bibliotecă selectat. Apasă Pornește.");
});

boot(() => {
  loadModels();
  initDashboard();
  schedule();
}, () => socket?.close());
