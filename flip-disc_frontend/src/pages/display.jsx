import { useCallback, useEffect, useRef, useState } from "react";
import { io } from "socket.io-client";
import { BACKEND_URL } from "../backend";
import CameraDialog from "../components/CameraDialog";
import FlipdotWebGL from "../components/Flipdot";
import MaskView from "../components/MaskView";

// The backend sends a frame in every mode, idle and camera-less included, so
// this long without one while connected means its pipeline is stuck. (Its
// watchdog restarts it after FLIPDISC_STALL_SECONDS, 10 s by default.)
const STALL_MS = 3000;

const VIEWS = [
  { id: "flipdot", label: "Flip-disc" },
  { id: "mask", label: "Mask" },
  { id: "camera", label: "Camera" },
];

const ICON_PATHS = {
  camera: "M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3zM15 13a3 3 0 1 1-6 0 3 3 0 0 1 6 0z",
  enter: "M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3",
  exit: "M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3",
};

// Fullscreen for whichever view is showing. In fullscreen the button and the
// cursor hide after a couple of seconds without mouse movement.
const useFullscreen = (ref) => {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [controlsVisible, setControlsVisible] = useState(true);

  const toggle = useCallback(() => {
    if (document.fullscreenElement) document.exitFullscreen();
    else ref.current?.requestFullscreen();
  }, [ref]);

  useEffect(() => {
    const onChange = () => setIsFullscreen(document.fullscreenElement === ref.current);
    const onKey = (e) => {
      if (e.target.closest?.("input, textarea")) return; // typing an F, not a shortcut
      if (e.key.toLowerCase() === "f" && !e.repeat && !e.ctrlKey && !e.metaKey && !e.altKey) toggle();
    };
    document.addEventListener("fullscreenchange", onChange);
    window.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("fullscreenchange", onChange);
      window.removeEventListener("keydown", onKey);
    };
  }, [ref, toggle]);

  useEffect(() => {
    setControlsVisible(true);
    if (!isFullscreen) return;
    let timer;
    const show = () => {
      setControlsVisible(true);
      clearTimeout(timer);
      timer = setTimeout(() => setControlsVisible(false), 2000);
    };
    show();
    window.addEventListener("mousemove", show);
    return () => {
      clearTimeout(timer);
      window.removeEventListener("mousemove", show);
    };
  }, [isFullscreen]);

  return { isFullscreen, controlsVisible, toggle };
};

// The backend sends the whole matrix once ("discs", a 0/1 string) and then only
// what changed ("on" / "off": flat disc indices). Unchanged frames keep the
// same `discs` array, so the views skip them without doing any work.
const applyUpdate = (prev, msg) => {
  const { discs, on, off, ...status } = msg;
  if (discs !== undefined) {
    return { ...status, discs: Uint8Array.from(discs, (ch) => (ch === "1" ? 1 : 0)) };
  }
  if (!prev) return null; // a delta before our snapshot: wait for the snapshot
  if (on.length === 0 && off.length === 0) return { ...status, discs: prev.discs };
  const next = prev.discs.slice();
  for (const i of on) next[i] = 1;
  for (const i of off) next[i] = 0;
  return { ...status, discs: next };
};

// Centred over the display so a lost connection can't go unnoticed.
const DisconnectedNotice = ({ attempt }) => (
  <div className="pointer-events-none absolute inset-0 flex items-center justify-center p-4">
    <div className="max-w-sm rounded-xl border border-red-500/40 bg-neutral-900/90 px-6 py-5 text-center shadow-2xl backdrop-blur">
      <div className="mb-2 flex items-center justify-center gap-2 text-base font-medium text-red-400">
        <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-red-500" />
        Disconnected
      </div>
      <p className="text-neutral-300">
        Lost the connection to the backend at <span className="font-mono">{BACKEND_URL}</span>.
      </p>
      <p className="mt-1 text-neutral-500">
        Reconnecting automatically{attempt > 0 && ` (attempt ${attempt})`}…
      </p>
    </div>
  </div>
);

// Centred too: the display is frozen meanwhile.
const StalledNotice = () => (
  <div className="pointer-events-none absolute inset-0 flex items-center justify-center p-4">
    <div className="max-w-sm rounded-xl border border-amber-500/40 bg-neutral-900/90 px-6 py-5 text-center shadow-2xl backdrop-blur">
      <div className="mb-2 flex items-center justify-center gap-2 text-base font-medium text-amber-400">
        <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-amber-400" />
        No frames from the backend
      </div>
      <p className="text-neutral-300">Connected, but the picture has stopped.</p>
      <p className="mt-1 text-neutral-500">The backend restarts itself if it stays stuck…</p>
    </div>
  </div>
);

// Top-left, out of the way: the idle clip is still worth watching meanwhile.
const NoCameraBadge = () => (
  <div
    title="The backend keeps retrying. Pick another camera with the camera button."
    className="absolute left-4 top-4 flex items-center gap-2 rounded-full border border-amber-500/40 bg-neutral-900/80 px-3 py-1 text-amber-400 backdrop-blur"
  >
    <span className="h-2 w-2 animate-pulse rounded-full bg-amber-400" />
    No camera signal
  </div>
);

const Status = ({ connected, frame }) => {
  if (!connected) return <span className="text-red-400">● disconnected</span>;
  if (!frame) return <span>● connected</span>;

  const idle = frame.mode === "idle_video";
  return (
    <span className="tabular-nums">
      <span className={idle ? "text-amber-400" : "text-green-400"}>●</span>{" "}
      {idle ? "idle" : "active"} · {frame.fps} fps · {frame.people} people
      {frame.nearest_m != null && ` · nearest ${frame.nearest_m} m`}
    </span>
  );
};

const Display = () => {
  const [view, setView] = useState("flipdot");
  const [connected, setConnected] = useState(false);
  // Set once a connection drops or a first attempt fails, not while the very
  // first connection is still pending, so the notice doesn't flash on load.
  const [lost, setLost] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const [session, setSession] = useState(0); // bumps on every (re)connect
  const [frame, setFrame] = useState(null);
  const [stalled, setStalled] = useState(false);
  const [cameraDialog, setCameraDialog] = useState(false);
  // Stable, or the dialog's effect (fetch + Esc handler) re-runs on every frame.
  const closeCameraDialog = useCallback(() => setCameraDialog(false), []);
  const mainRef = useRef(null);
  const { isFullscreen, controlsVisible, toggle } = useFullscreen(mainRef);

  useEffect(() => {
    const socket = io(BACKEND_URL);
    socket.on("connect", () => {
      setConnected(true);
      setLost(false);
      setAttempt(0);
      setSession((n) => n + 1);
    });
    // When the last frame came; null until the first one of this connection.
    let lastUpdate = null;
    socket.on("disconnect", () => {
      setConnected(false);
      setLost(true);
      setFrame(null);
      lastUpdate = null;
    });
    socket.on("connect_error", () => setLost(true));
    socket.io.on("reconnect_attempt", setAttempt);
    socket.on("flipdisc_update", (msg) => {
      lastUpdate = performance.now();
      setFrame((prev) => applyUpdate(prev, msg));
    });
    const stallCheck = setInterval(() => {
      setStalled(lastUpdate !== null && performance.now() - lastUpdate > STALL_MS);
    }, 1000);
    return () => {
      clearInterval(stallCheck);
      socket.disconnect();
    };
  }, []);

  return (
    <div className="flex h-dvh flex-col bg-[#1a1a1a] text-sm text-neutral-300">
      <header className="flex flex-wrap items-center justify-between gap-2 px-3 py-2">
        <nav className="flex rounded-lg bg-neutral-800 p-1">
          {VIEWS.map(({ id, label }) => (
            <button
              key={id}
              onClick={() => setView(id)}
              className={`rounded-md px-3 py-1 transition-colors ${
                view === id ? "bg-neutral-600 text-white" : "hover:text-white"
              }`}
            >
              {label}
            </button>
          ))}
        </nav>
        <div className="flex items-center gap-3">
          <Status connected={connected} frame={frame} />
          <button
            onClick={() => setCameraDialog(true)}
            title="Camera"
            aria-label="Choose camera"
            className="rounded-lg bg-neutral-800 p-1.5 text-neutral-300 transition-colors hover:bg-neutral-700 hover:text-white"
          >
            <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor"
                 strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d={ICON_PATHS.camera} />
            </svg>
          </button>
        </div>
      </header>

      <main
        ref={mainRef}
        className={`relative min-h-0 flex-1 bg-[#1a1a1a] ${isFullscreen ? "" : "p-2"} ${
          controlsVisible ? "" : "cursor-none"
        }`}
      >
        {view === "flipdot" && <FlipdotWebGL discs={frame?.discs} />}
        {view === "mask" && <MaskView discs={frame?.discs} />}
        {/* Only streams (and only costs the backend a JPEG encode) while shown. */}
        {view === "camera" && (
          <img
            key={session} // a dropped MJPEG stream doesn't resume by itself
            src={`${BACKEND_URL}/debug.mjpg`}
            alt="Camera with silhouette and distance boxes"
            className="block h-full w-full object-contain"
          />
        )}
        {lost && <DisconnectedNotice attempt={attempt} />}
        {!lost && stalled && <StalledNotice />}
        {!lost && !stalled && frame?.camera_ok === false && <NoCameraBadge />}
        {document.fullscreenEnabled && (
          <button
            onClick={toggle}
            title={isFullscreen ? "Exit fullscreen (F)" : "Fullscreen (F)"}
            aria-label={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
            className={`absolute bottom-4 right-4 rounded-lg bg-neutral-800/80 p-2 text-neutral-300 transition-opacity hover:bg-neutral-700 hover:text-white ${
              controlsVisible ? "opacity-100" : "pointer-events-none opacity-0"
            }`}
          >
            <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor"
                 strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d={ICON_PATHS[isFullscreen ? "exit" : "enter"]} />
            </svg>
          </button>
        )}
      </main>
      {cameraDialog && <CameraDialog onClose={closeCameraDialog} />}
    </div>
  );
};

export default Display;
