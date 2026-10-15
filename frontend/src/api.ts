import { showLogin } from "./auth";

export const modelLabel = (path: string) => (path.endsWith("best.pt") ? path.split(/[\\/]/).at(-3) ?? path : path);

export async function fetchModels(): Promise<string[] | null> {
  const res = await fetch("/api/models");
  if (res.status === 401) {
    showLogin("Sesiunea a expirat");
    return null;
  }
  return ((await res.json()) as { models: string[] }).models;
}

export function upload(file: File, onProgress: (percent: number) => void): Promise<string> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append("file", file);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(Math.round((e.loaded / e.total) * 100));
    xhr.onload = () => {
      let body: { id?: string; detail?: string } = {};
      try {
        body = JSON.parse(xhr.responseText);
      } catch {}
      if (xhr.status === 401) showLogin("Sesiunea a expirat");
      if (xhr.status === 413) return reject(new Error("Fișier prea mare pentru upload (max. 100 MB prin tunel). Pune-l în Bibliotecă."));
      if (xhr.status === 200 && body.id) resolve(body.id);
      else reject(new Error(body.detail ?? `Încărcarea a eșuat (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error("Încărcarea a eșuat"));
    xhr.open("POST", "/api/videos");
    xhr.send(form);
  });
}
