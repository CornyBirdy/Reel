async function request(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!res.ok) {
    const detail = data && typeof data === "object" ? data.detail : data;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail) || res.statusText);
  }
  return data;
}

// A picture or video chosen on this PC: the file is the request body (see POST /api/inputs).
async function upload(file) {
  const res = await fetch("/api/inputs", { method: "POST", headers: { "Content-Type": file.type || "application/octet-stream" }, body: file });
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!res.ok) throw new Error((data && data.detail) || `Upload failed (${res.status}).`);
  return data;
}

export const api = {
  upload,
  get: (p) => request("GET", p),
  post: (p, b) => request("POST", p, b ?? {}),
  put: (p, b) => request("PUT", p, b ?? {}),
  del: (p) => request("DELETE", p),
};
