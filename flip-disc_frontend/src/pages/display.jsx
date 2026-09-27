import { useCallback, useEffect, useRef, useState } from "react";
import { io } from "socket.io-client";
import FlipdotWebGL from "../components/Flipdot";
import MaskView from "../components/MaskView";

// Same host the page came from, so the display also works from another device on the LAN.
const BACKEND_URL = `http://${window.location.hostname}:5000`;

const VIEWS = [
  { id: "flipdot", label: "Flip-disc" },
  { id: "mask", label: "Mask" },
  { id: "camera", label: "Camera" },
];

const ICON_PATHS = {
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
  const [frame, setFrame] = useState(null);
  const mainRef = useRef(null);
  const { isFullscreen, controlsVisible, toggle } = useFullscreen(mainRef);

  useEffect(() => {
    const socket = io(BACKEND_URL);
    socket.on("connect", () => setConnected(true));
    socket.on("disconnect", () => {
      setConnected(false);
      setFrame(null);
    });
    socket.on("flipdisc_update", setFrame);
    return () => socket.disconnect();
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
        <Status connected={connected} frame={frame} />
      </header>

      <main
        ref={mainRef}
        className={`relative min-h-0 flex-1 bg-[#1a1a1a] ${isFullscreen ? "" : "p-2"} ${
          controlsVisible ? "" : "cursor-none"
        }`}
      >
        {view === "flipdot" && <FlipdotWebGL matrix={frame?.matrix} />}
        {view === "mask" && <MaskView matrix={frame?.matrix} />}
        {/* Only streams (and only costs the backend a JPEG encode) while shown. */}
        {view === "camera" && (
          <img
            src={`${BACKEND_URL}/debug.mjpg`}
            alt="Camera with silhouette and distance boxes"
            className="block h-full w-full object-contain"
          />
        )}
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
    </div>
  );
};

export default Display;
