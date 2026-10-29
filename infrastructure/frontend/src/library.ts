import { showLogin } from "./auth";
import { $ } from "./dom";

interface LibraryVideo {
  name: string;
  size: number;
  modified: number;
  width: number;
  height: number;
  fps: number;
  duration: number;
}

const rows = $<HTMLTableSectionElement>("library-rows");
const empty = $("library-empty");
const views = { analysis: $("tab-analysis"), library: $("tab-library") };

const mb = (bytes: number) => (bytes >= 1e9 ? `${(bytes / 1e9).toFixed(2)} GB` : `${(bytes / 1e6).toFixed(1)} MB`);
const duration = (s: number) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;
const date = (t: number) => new Date(t * 1000).toLocaleDateString("ro-RO", { day: "2-digit", month: "short", year: "numeric" });

let onPick: (name: string) => void = () => {};

export function showTab(tab: keyof typeof views): void {
  for (const [key, el] of Object.entries(views)) el.hidden = key !== tab;
  document.querySelectorAll<HTMLButtonElement>(".tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === tab));
  if (tab === "library") load();
  window.dispatchEvent(new Event("resize"));
}

function row(v: LibraryVideo): HTMLTableRowElement {
  const tr = document.createElement("tr");
  const thumb = document.createElement("img");
  thumb.loading = "lazy";
  thumb.alt = "";
  thumb.src = `/api/library/${encodeURIComponent(v.name)}/thumbnail`;
  const pick = document.createElement("button");
  pick.type = "button";
  pick.textContent = "Analizează";
  pick.addEventListener("click", () => {
    onPick(v.name);
    showTab("analysis");
  });
  const cells = [thumb, v.name, `${v.width}×${v.height}`, duration(v.duration), mb(v.size), date(v.modified), pick];
  for (const content of cells) {
    const td = document.createElement("td");
    td.append(content);
    tr.append(td);
  }
  tr.children[1].className = "name";
  return tr;
}

async function load(): Promise<void> {
  const res = await fetch("/api/library");
  if (res.status === 401) return showLogin("Sesiunea a expirat");
  const { videos } = (await res.json()) as { videos: LibraryVideo[] };
  rows.replaceChildren(...videos.map(row));
  empty.hidden = videos.length > 0;
}

export function initLibrary(pick: (name: string) => void): void {
  onPick = pick;
  document.querySelectorAll<HTMLButtonElement>(".tabs button").forEach((b) =>
    b.addEventListener("click", () => showTab(b.dataset.tab as keyof typeof views)),
  );
  $("library-refresh").addEventListener("click", load);
}
