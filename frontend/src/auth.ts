import { $ } from "./dom";

const login = $("login");
const app = $("app");
const form = $<HTMLFormElement>("login-form");
const code = $<HTMLInputElement>("code");
const message = $("login-message");
const logout = $<HTMLButtonElement>("logout");

let enabled = false;
let onReady: () => void = () => {};

export function showLogin(text = ""): void {
  app.hidden = true;
  login.hidden = false;
  message.textContent = text;
  code.value = "";
  code.focus();
}

function showApp(): void {
  login.hidden = true;
  app.hidden = false;
  logout.hidden = !enabled;
  onReady();
}

async function submit(): Promise<void> {
  const res = await fetch("/api/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code: code.value.trim() }),
  });
  if (res.ok) return showApp();
  const body = (await res.json().catch(() => ({}))) as { detail?: string };
  showLogin(body.detail ?? "Autentificare eșuată");
}

export async function boot(ready: () => void, onLogout: () => void): Promise<void> {
  onReady = ready;
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    submit();
  });
  code.addEventListener("input", () => {
    code.value = code.value.replace(/\D/g, "").slice(0, 6);
    if (code.value.length === 6) submit();
  });
  logout.addEventListener("click", async () => {
    onLogout();
    await fetch("/api/logout", { method: "POST" });
    showLogin();
  });
  try {
    const state = (await (await fetch("/api/session")).json()) as { enabled: boolean; authenticated: boolean };
    enabled = state.enabled;
    if (enabled && !state.authenticated) return showLogin();
  } catch {}
  showApp();
}
