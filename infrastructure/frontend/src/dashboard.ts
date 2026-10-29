import { fit, flowField, sparkline, stackedArea } from "./charts";
import { $ } from "./dom";
import type { Meta, Metrics } from "./types";

const ui = {
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

const LIVE = 1;

let meta: Meta | null = null;
let latest: Metrics | null = null;
let frame: ImageBitmap | null = null;
let congestionUrl: string | null = null;
let classKeys = "";
let scheduled = false;

const color = (id: number) => meta?.colors[id] ?? "#a1a1aa";
const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
const clock = (s: number) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

export function schedule(): void {
  if (scheduled) return;
  scheduled = true;
  requestAnimationFrame(draw);
}

export function resetDashboard(): void {
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

export function initDashboard(): void {
  draw();
  flowField(ui.flow, [], 16 / 9, color);
  stackedArea(ui.timeline, [], [], 10);
  window.addEventListener("resize", schedule);
}

export function showMeta(m: Meta): void {
  meta = m;
  ui.statusModel.textContent = `${m.model} · ${m.device}`;
  ui.liveNote.textContent = `sursă ${m.width}×${m.height} · ${Math.round(m.fps)} fps`;
  ui.placeholder.hidden = true;
  ui.tracked.hidden = !m.track;
}

export function showMetrics(m: Metrics): void {
  latest = m;
  schedule();
}

export function showDownloads(files: Record<string, string>): void {
  ui.dlVideo.href = files["detectii.mp4"] ?? "#";
  ui.dlCsv.href = files["numaratoare.csv"] ?? "#";
  ui.downloads.hidden = !Object.keys(files).length;
}

export async function showImage(buf: ArrayBuffer): Promise<void> {
  const blob = new Blob([new Uint8Array(buf, 1)], { type: "image/jpeg" });
  if (new Uint8Array(buf, 0, 1)[0] === LIVE) {
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
    (ui.share.children[i] as HTMLElement).style.flexGrow = String(c.share);
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
