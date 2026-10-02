import type { FlowVector } from "./types";

const MUTED = "#8c887c";
const GRID = "#e3dfd3";
const FIELD = "#ebe8df";
const MONO = '10px "IBM Plex Mono", monospace';

export function fit(canvas: HTMLCanvasElement): [CanvasRenderingContext2D, number, number] {
  const dpr = window.devicePixelRatio || 1;
  const { width, height } = canvas.getBoundingClientRect();
  const w = Math.round(width * dpr);
  const h = Math.round(height * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return [ctx, width, height];
}

export function sparkline(canvas: HTMLCanvasElement, values: number[], color: string): void {
  const [ctx, w, h] = fit(canvas);
  if (values.length < 2) return;
  const pad = 3;
  const max = Math.max(1, ...values);
  const x = (i: number) => pad + (i / (values.length - 1)) * (w - pad * 2);
  const y = (v: number) => h - pad - (v / max) * (h - pad * 2);
  ctx.beginPath();
  ctx.moveTo(x(0), y(values[0]));
  for (let i = 1; i < values.length; i++) {
    ctx.lineTo(x(i), y(values[i - 1]));
    ctx.lineTo(x(i), y(values[i]));
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.2;
  ctx.stroke();
  ctx.lineTo(x(values.length - 1), h - pad);
  ctx.lineTo(x(0), h - pad);
  ctx.globalAlpha = 0.16;
  ctx.fillStyle = color;
  ctx.fill();
  ctx.globalAlpha = 1;
  ctx.beginPath();
  ctx.arc(x(values.length - 1), y(values.at(-1)!), 2, 0, Math.PI * 2);
  ctx.fill();
}

function niceMax(value: number): number {
  return Math.max(4, Math.ceil(value / 4) * 4);
}

export function stackedArea(canvas: HTMLCanvasElement, series: number[][], colors: string[], seconds: number): void {
  const [ctx, w, h] = fit(canvas);
  const left = 2, right = 30, top = 6, bottom = 18;
  const pw = w - left - right;
  const ph = h - top - bottom;
  const n = series[0]?.length ?? 0;
  const totals = Array.from({ length: n }, (_, i) => series.reduce((s, row) => s + row[i], 0));
  const max = niceMax(Math.max(0, ...totals));
  const x = (i: number) => left + (n > 1 ? (i / (n - 1)) * pw : pw);
  const y = (v: number) => top + ph - (v / max) * ph;

  ctx.font = MONO;
  ctx.fillStyle = MUTED;
  ctx.strokeStyle = GRID;
  ctx.lineWidth = 1;
  for (let k = 0; k <= 4; k++) {
    const v = (max / 4) * k;
    ctx.beginPath();
    ctx.moveTo(left, Math.round(y(v)) + 0.5);
    ctx.lineTo(left + pw, Math.round(y(v)) + 0.5);
    ctx.stroke();
    ctx.fillText(String(Math.round(v)), left + pw + 6, y(v) + 3);
  }
  ctx.fillText(`-${Math.round(seconds)}s`, left, h - 4);
  ctx.textAlign = "right";
  ctx.fillText("acum", left + pw, h - 4);
  ctx.textAlign = "left";
  if (n < 2) return;

  const base = new Array<number>(n).fill(0);
  series.forEach((row, s) => {
    const upper = row.map((v, i) => base[i] + v);
    ctx.beginPath();
    upper.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
    for (let i = n - 1; i >= 0; i--) ctx.lineTo(x(i), y(base[i]));
    ctx.closePath();
    ctx.globalAlpha = 0.5;
    ctx.fillStyle = colors[s];
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.beginPath();
    upper.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
    ctx.strokeStyle = colors[s];
    ctx.lineWidth = 1;
    ctx.stroke();
    upper.forEach((v, i) => (base[i] = v));
  });
  ctx.beginPath();
  ctx.arc(x(n - 1), y(totals[n - 1]), 3, 0, Math.PI * 2);
  ctx.fillStyle = "#232320";
  ctx.fill();
}

export function flowField(canvas: HTMLCanvasElement, flow: FlowVector[], aspect: number, color: (cls: number) => string): void {
  const [ctx, w, h] = fit(canvas);
  const rw = Math.min(w, h * aspect);
  const rh = rw / aspect;
  const rx = (w - rw) / 2;
  const ry = (h - rh) / 2;
  ctx.fillStyle = FIELD;
  ctx.fillRect(rx, ry, rw, rh);
  ctx.strokeStyle = GRID;
  ctx.lineWidth = 1;
  for (let k = 1; k < 8; k++) {
    ctx.beginPath();
    ctx.moveTo(rx + (rw * k) / 8, ry);
    ctx.lineTo(rx + (rw * k) / 8, ry + rh);
    ctx.moveTo(rx, ry + (rh * k) / 8);
    ctx.lineTo(rx + rw, ry + (rh * k) / 8);
    ctx.stroke();
  }
  const horizon = 12;
  for (const [fx, fy, vx, vy, cls] of flow) {
    const x0 = rx + fx * rw;
    const y0 = ry + fy * rh;
    const dx = vx * rw * horizon;
    const dy = vy * rh * horizon;
    const c = color(cls);
    ctx.fillStyle = c;
    ctx.strokeStyle = c;
    ctx.beginPath();
    ctx.arc(x0, y0, 1.8, 0, Math.PI * 2);
    ctx.fill();
    const len = Math.hypot(dx, dy);
    if (len < 2) continue;
    const a = Math.atan2(dy, dx);
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x0 + dx, y0 + dy);
    ctx.moveTo(x0 + dx, y0 + dy);
    ctx.lineTo(x0 + dx - 5 * Math.cos(a - 0.45), y0 + dy - 5 * Math.sin(a - 0.45));
    ctx.moveTo(x0 + dx, y0 + dy);
    ctx.lineTo(x0 + dx - 5 * Math.cos(a + 0.45), y0 + dy - 5 * Math.sin(a + 0.45));
    ctx.stroke();
  }
}
