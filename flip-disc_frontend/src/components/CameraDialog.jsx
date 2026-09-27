import { useCallback, useEffect, useState } from "react";
import { api } from "../backend";

// The camera manager: pick the active camera, and add, edit or delete saved
// ones (webcams and IP cameras). The backend keeps the list in cameras.json,
// never sends passwords back, and only switches to a camera that delivers a
// picture.

const RESOLUTIONS = [[640, 360], [960, 540], [1280, 720], [1920, 1080]];

// Starting values for a new camera: a guide, all editable.
const DEFAULTS = {
  webcam: { name: "Webcam", index: 0, resolution: [1280, 720] },
  ip: {
    name: "IP camera", brand: "dahua", ip: "", port: 554, username: "admin", password: "",
    channel: 1, stream: "main", resolution: [1280, 720],
  },
};

// Mirrors camera_url() in the backend, for showing the URL as it's typed.
const previewUrl = (f) => {
  const login = f.username ? `${f.username}${f.password || f.hasPassword ? ":***" : ""}@` : "";
  const base = `rtsp://${login}${f.ip || "<ip>"}:${f.port}`;
  return f.brand === "hikvision"
    ? `${base}/Streaming/Channels/${f.channel}0${f.stream === "main" ? 1 : 2}`
    : `${base}/cam/realmonitor?channel=${f.channel}&subtype=${f.stream === "main" ? 0 : 1}`;
};

// Form state -> what the backend expects for that camera type.
const payload = (f) =>
  f.type === "webcam"
    ? { type: "webcam", name: f.name, index: f.index, resolution: f.resolution }
    : {
        type: "ip", name: f.name, brand: f.brand, ip: f.ip.trim(), port: Number(f.port),
        username: f.username, password: f.password, channel: Number(f.channel),
        stream: f.stream, resolution: f.resolution,
      };

// ---- small building blocks ----------------------------------------------------

const Button = ({ variant = "plain", className = "", ...props }) => {
  const look = {
    plain: "bg-neutral-700 text-neutral-100 hover:bg-neutral-600",
    primary: "bg-emerald-600 text-white hover:bg-emerald-500",
    danger: "bg-red-600 text-white hover:bg-red-500",
    ghost: "text-neutral-300 hover:bg-neutral-800 hover:text-white",
  }[variant];
  return (
    <button
      {...props}
      className={`rounded-md px-3 py-1.5 transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${look} ${className}`}
    />
  );
};

const inputClass =
  "w-full rounded-md border border-neutral-700 bg-neutral-950 px-3 py-1.5 text-neutral-100 " +
  "placeholder:text-neutral-600 focus:border-emerald-500 focus:outline-none";

const Field = ({ label, hint, children }) => (
  <label className="block">
    <span className="mb-1 block text-xs text-neutral-400">{label}</span>
    {children}
    {hint && <span className="mt-1 block text-xs text-neutral-500">{hint}</span>}
  </label>
);

const Select = ({ value, onChange, options }) => (
  <select value={value} onChange={(e) => onChange(e.target.value)} className={inputClass}>
    {options.map(([v, text]) => (
      <option key={v} value={v}>{text}</option>
    ))}
  </select>
);

const ResolutionSelect = ({ value, onChange }) => (
  <Select
    value={value.join("x")}
    onChange={(v) => onChange(v.split("x").map(Number))}
    options={RESOLUTIONS.map(([w, h]) => [`${w}x${h}`, `${w} × ${h}`])}
  />
);

const Thumbnail = ({ src, className = "w-32" }) => (
  <div className={`aspect-video shrink-0 overflow-hidden rounded-md bg-black ${className}`}>
    {src && <img src={src} alt="" className="h-full w-full object-cover" />}
  </div>
);

const TypeBadge = ({ type }) => (
  <span className="rounded bg-neutral-700 px-1.5 py-0.5 text-[11px] uppercase tracking-wide text-neutral-300">
    {type === "ip" ? "IP" : "Webcam"}
  </span>
);

// ---- the add / edit form --------------------------------------------------------

const CameraForm = ({ editing, run, onSaved, onCancel }) => {
  // One state holding both types' fields, so switching type back and forth
  // doesn't lose what was typed.
  const [form, setForm] = useState(() => {
    const start = { type: editing?.type ?? "webcam", ...DEFAULTS.webcam, ...DEFAULTS.ip };
    if (!editing) return { ...start, name: DEFAULTS.webcam.name };
    return { ...start, ...editing, password: "", hasPassword: !!editing.has_password };
  });
  const [webcams, setWebcams] = useState(null);
  const [tested, setTested] = useState(null);
  const set = (changes) => {
    setForm((f) => ({ ...f, ...changes }));
    setTested(null);
  };

  const setType = (type) => {
    // Swap in the other type's default name, unless the user typed their own.
    const defaultNames = [DEFAULTS.webcam.name, DEFAULTS.ip.name];
    set({ type, ...(defaultNames.includes(form.name) || /^Webcam \d+$/.test(form.name)
      ? { name: DEFAULTS[type].name } : {}) });
  };

  const scan = () =>
    run("Looking for webcams…", async () => {
      const r = await api("/cameras/scan");
      setWebcams(r.webcams);
    });

  const test = () =>
    run(form.type === "ip" ? "Testing the IP camera (up to ~10 s)…" : "Testing the webcam…", async () => {
      setTested(await api("/cameras/test", { ...payload(form), id: editing?.id }));
    });

  const save = () =>
    run(editing ? "Saving…" : "Adding the camera…", async () => {
      const r = editing
        ? await api(`/cameras/${editing.id}`, payload(form), "PUT")
        : await api("/cameras", payload(form));
      if (!r.ok) throw new Error(r.error);
      onSaved(r, editing ? `Saved '${form.name}'.` : `Added '${form.name}'. Pick it above to use it.`);
    });

  return (
    <div className="space-y-4">
      <div className="flex rounded-lg bg-neutral-800 p-1">
        {[["webcam", "Webcam"], ["ip", "IP camera"]].map(([type, label]) => (
          <button
            key={type}
            onClick={() => setType(type)}
            className={`flex-1 rounded-md px-3 py-1 transition-colors ${
              form.type === type ? "bg-neutral-600 text-white" : "hover:text-white"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {form.type === "webcam" ? (
        <>
          <Field label="Device" hint="Scan for the webcams plugged into the backend machine, then pick one.">
            {webcams === null ? (
              <div className="text-neutral-400">Webcam {form.index}</div>
            ) : webcams.length === 0 ? (
              <div className="text-neutral-500">No webcams found.</div>
            ) : (
              <div className="grid grid-cols-2 gap-2">
                {webcams.map((w) => (
                  <button
                    key={w.index}
                    onClick={() => set({
                      index: w.index,
                      ...(/^Webcam( \d+)?$/.test(form.name) ? { name: `Webcam ${w.index}` } : {}),
                    })}
                    className={`rounded-lg border p-1.5 text-left transition-colors ${
                      form.index === w.index ? "border-emerald-500 bg-emerald-500/10" : "border-neutral-700 hover:border-neutral-500"
                    }`}
                  >
                    <Thumbnail src={w.thumbnail} className="w-full" />
                    <div className="mt-1 flex items-center justify-between text-xs">
                      <span className="text-white">Webcam {w.index}</span>
                      {w.in_use && <span className="text-green-400">in use</span>}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </Field>
          <Button onClick={scan}>{webcams === null ? "Scan" : "Scan again"}</Button>
          <Field label="Name">
            <input className={inputClass} value={form.name} maxLength={40} onChange={(e) => set({ name: e.target.value })} />
          </Field>
          <Field label="Resolution">
            <ResolutionSelect value={form.resolution} onChange={(resolution) => set({ resolution })} />
          </Field>
        </>
      ) : (
        <>
          <Field label="Name">
            <input className={inputClass} value={form.name} maxLength={40} onChange={(e) => set({ name: e.target.value })} />
          </Field>
          <div className="grid grid-cols-3 gap-3">
            <div className="col-span-2">
              <Field label="IP address">
                <input className={`${inputClass} font-mono`} value={form.ip} placeholder="192.168.1.188"
                       spellCheck={false} onChange={(e) => set({ ip: e.target.value })} />
              </Field>
            </div>
            <Field label="Port">
              <input className={`${inputClass} font-mono`} type="number" min={1} max={65535} value={form.port}
                     onChange={(e) => set({ port: e.target.value })} />
            </Field>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Username">
              <input className={inputClass} value={form.username} autoComplete="off" spellCheck={false}
                     onChange={(e) => set({ username: e.target.value })} />
            </Field>
            <Field label="Password">
              <input className={inputClass} type="password" value={form.password} autoComplete="new-password"
                     placeholder={form.hasPassword ? "unchanged" : ""}
                     onChange={(e) => set({ password: e.target.value })} />
            </Field>
          </div>
          <div className="grid grid-cols-3 gap-3">
            <Field label="Brand">
              <Select value={form.brand} onChange={(brand) => set({ brand })}
                      options={[["dahua", "Dahua"], ["hikvision", "Hikvision"]]} />
            </Field>
            <Field label="Channel">
              <input className={inputClass} type="number" min={1} max={999} value={form.channel}
                     onChange={(e) => set({ channel: e.target.value })} />
            </Field>
            <Field label="Stream">
              <Select value={form.stream} onChange={(stream) => set({ stream })}
                      options={[["main", "Main"], ["sub", "Sub"]]} />
            </Field>
          </div>
          <Field label="Resolution" hint="The stream is scaled to this. A sub stream is usually 640 × 360 or smaller.">
            <ResolutionSelect value={form.resolution} onChange={(resolution) => set({ resolution })} />
          </Field>
          <div className="break-all rounded-md bg-neutral-950 px-3 py-2 font-mono text-xs text-neutral-400">
            {previewUrl(form)}
          </div>
        </>
      )}

      {tested && (
        <div className="flex items-center gap-3 rounded-lg bg-neutral-800 p-2">
          {tested.ok && <Thumbnail src={tested.thumbnail} />}
          <div className={tested.ok ? "text-green-400" : "text-red-400"}>
            {tested.ok
              ? tested.in_use ? "Works — it's the camera running now." : `Works — the camera sends ${tested.width} × ${tested.height}.`
              : `Doesn't work: ${tested.error}`}
          </div>
        </div>
      )}

      <div className="flex justify-between gap-2">
        <Button variant="ghost" onClick={onCancel}>Cancel</Button>
        <div className="flex gap-2">
          <Button onClick={test} disabled={form.type === "ip" && !form.ip.trim()}>Test</Button>
          <Button variant="primary" onClick={save} disabled={!form.name.trim() || (form.type === "ip" && !form.ip.trim())}>
            {editing ? "Save" : "Add"}
          </Button>
        </div>
      </div>
    </div>
  );
};

// ---- the dialog -------------------------------------------------------------------

const CameraDialog = ({ onClose }) => {
  const [list, setList] = useState(null); // { active, cameras, signal }
  const [mode, setMode] = useState("list"); // "list" | "add" | camera being edited
  const [busy, setBusy] = useState(null);
  const [message, setMessage] = useState(null); // { ok, text }
  const [confirmDelete, setConfirmDelete] = useState(null);

  useEffect(() => {
    api("/cameras")
      .then(setList)
      .catch(() => setMessage({ ok: false, text: "Can't reach the backend." }));
  }, []);

  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Every action goes through here: one thing at a time, its label shown while
  // it runs, and its error (thrown or from the backend) shown after.
  const run = useCallback(async (label, work) => {
    setBusy(label);
    setMessage(null);
    try {
      await work();
    } catch (e) {
      setMessage({ ok: false, text: e instanceof TypeError ? "Can't reach the backend." : e.message });
    } finally {
      setBusy(null);
    }
  }, []);

  const apply = (r, text) => {
    setList(r);
    setMode("list");
    setMessage({ ok: true, text });
  };

  const activate = (id) =>
    run("Switching camera (up to ~10 s)…", async () => {
      const r = await api("/cameras/active", { id });
      if (!r.ok) throw new Error(r.error);
      apply(r, `Now using '${r.cameras.find((c) => c.id === id).name}'.`);
    });

  const remove = (cam) =>
    run("Deleting…", async () => {
      const r = await api(`/cameras/${cam.id}`, undefined, "DELETE");
      if (!r.ok) throw new Error(r.error);
      setConfirmDelete(null);
      apply(r, `Deleted '${cam.name}'.`);
    });

  const active = list?.cameras.find((c) => c.id === list.active);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      onMouseDown={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="camera-dialog-title"
        className="max-h-full w-full max-w-lg overflow-y-auto rounded-xl border border-neutral-700 bg-neutral-900 p-5 text-sm text-neutral-300 shadow-2xl"
      >
        <div className="mb-4 flex items-center justify-between">
          <h2 id="camera-dialog-title" className="text-base font-medium text-white">
            {mode === "list" ? "Cameras" : mode === "add" ? "Add camera" : `Edit '${mode.name}'`}
          </h2>
          <button onClick={onClose} aria-label="Close" className="rounded p-1 text-neutral-400 hover:text-white">
            <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M18 6 6 18M6 6l12 12" />
            </svg>
          </button>
        </div>

        <fieldset disabled={!!busy} className="min-w-0">
          {mode !== "list" ? (
            <CameraForm
              editing={mode === "add" ? null : mode}
              run={run}
              onSaved={apply}
              onCancel={() => setMode("list")}
            />
          ) : !list ? (
            <p className="text-neutral-500">Loading…</p>
          ) : (
            <div className="space-y-4">
              <Field label="Camera in use">
                <div className="flex items-center gap-2">
                  <span className={`h-2 w-2 shrink-0 rounded-full ${list.signal ? "bg-green-400" : "bg-amber-400"}`}
                        title={list.signal ? "Picture coming in" : "No picture"} />
                  <select value={list.active} onChange={(e) => activate(Number(e.target.value))} className={inputClass}>
                    {list.cameras.map((c) => (
                      <option key={c.id} value={c.id}>
                        {c.name} — {c.type === "ip" ? "IP camera" : "webcam"}
                      </option>
                    ))}
                  </select>
                </div>
                {!list.signal && active && (
                  <span className="mt-1 block text-xs text-amber-400">No picture from {active.name} — the backend keeps retrying.</span>
                )}
              </Field>

              <ul className="space-y-2">
                {list.cameras.map((cam) => (
                  <li key={cam.id} className="rounded-lg bg-neutral-800 p-3">
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="truncate text-white">{cam.name}</span>
                          <TypeBadge type={cam.type} />
                          {cam.id === list.active && <span className="text-xs text-green-400">in use</span>}
                        </div>
                        <div className="mt-0.5 text-xs text-neutral-400">{cam.summary}</div>
                      </div>
                      <div className="flex shrink-0 gap-1">
                        <Button variant="ghost" onClick={() => { setMessage(null); setMode(cam); }}>Edit</Button>
                        {confirmDelete === cam.id ? (
                          <Button variant="danger" onClick={() => remove(cam)} onBlur={() => setConfirmDelete(null)} autoFocus>
                            Delete?
                          </Button>
                        ) : (
                          <Button
                            variant="ghost"
                            disabled={cam.id === list.active}
                            title={cam.id === list.active ? "Switch to another camera first" : undefined}
                            onClick={() => setConfirmDelete(cam.id)}
                          >
                            Delete
                          </Button>
                        )}
                      </div>
                    </div>
                  </li>
                ))}
              </ul>

              <Button variant="primary" onClick={() => { setMessage(null); setMode("add"); }}>+ Add camera</Button>
            </div>
          )}
        </fieldset>

        {(busy || message) && (
          <div
            className={`mt-4 rounded-md px-3 py-2 ${
              busy ? "bg-neutral-800 text-neutral-300" : message.ok ? "bg-green-500/10 text-green-400" : "bg-red-500/10 text-red-400"
            }`}
          >
            {busy || message.text}
          </div>
        )}
      </div>
    </div>
  );
};

export default CameraDialog;
