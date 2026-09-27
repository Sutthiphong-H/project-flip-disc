// Built, the page is served by the backend itself; under `npm run dev` it comes
// from Vite, and the backend is on port 5000 of the same host (so a phone on
// the LAN works too).
export const BACKEND_URL = import.meta.env.DEV
  ? `http://${window.location.hostname}:5000`
  : window.location.origin;

// GET without a body, POST (or `method`) with JSON. Resolves to the parsed JSON
// even on an HTTP error status: the backend puts the reason in `error`.
export const api = async (path, body, method) => {
  const options = { method: method ?? (body === undefined ? "GET" : "POST") };
  if (body !== undefined) {
    options.headers = { "Content-Type": "application/json" };
    options.body = JSON.stringify(body);
  }
  const response = await fetch(BACKEND_URL + path, options);
  return response.json();
};
