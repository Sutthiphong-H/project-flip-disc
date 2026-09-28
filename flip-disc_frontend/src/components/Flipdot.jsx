import { useEffect, useRef, useState } from "react";
import createREGL from "regl";

export const ROWS = 45;
export const COLS = 80;
const COUNT = ROWS * COLS;
// One flip as a coil drives it: the disc starts slowly and speeds up until it
// hits its stop (TURN_MS), then bounces back twice, less each time (SETTLE_MS).
// The turn has to span several screen frames to read as a turn at all -- 90 ms
// is ~5 at 60 Hz. (An 80 ms ease-out turned it in ~24 ms: a blink, not a flip.)
const TURN_MS = 90;
const SETTLE_MS = 70;
const FLIP_MS = TURN_MS + SETTLE_MS;
const TURN_SHARE = TURN_MS / FLIP_MS;
const SWEEP_MS = 40; // a real controller drives column by column: left edge flips first
const DOT_SIZE = 0.92; // disc diameter as a fraction of the grid pitch

// Leaf-inspired color palette
const leafColors = [
  [0.20, 0.76, 0.25], // Fresh green
  [0.13, 0.55, 0.13], // Medium green
  [0.00, 0.39, 0.00], // Dark green
  [0.56, 0.93, 0.56], // Light green
  [0.48, 0.77, 0.46], // Seafoam green
  [0.67, 0.84, 0.29], // Yellow-green
  [0.33, 0.42, 0.18], // Olive green
  [0.19, 0.50, 0.08], // Forest green
  [0.38, 0.61, 0.25], // Grass green
  [0.29, 0.70, 0.33]  // Spring green
];

const randomLeafColor = () =>
  leafColors[Math.floor(Math.random() * leafColors.length)].map(
    (v) => v * (0.9 + Math.random() * 0.2)
  );

const randomFlips = (flips) => {
  for (let i = 0; i < COUNT; i++) flips[i] = Math.random() > 0.7 ? 1 : 0;
};

// Dot centres in grid space, -1..1 on both axes. The vertex shader scales this
// into a letterboxed 80:45 area, so the grid keeps its shape at any window size.
const positions = new Float32Array(COUNT * 2);
const sweepDelay = new Float32Array(COUNT); // ms after a frame lands before this disc starts
for (let i = 0; i < COUNT; i++) {
  const row = Math.floor(i / COLS);
  const col = i % COLS;
  positions[i * 2] = ((col + 0.5) / COLS) * 2 - 1;
  positions[i * 2 + 1] = 1 - ((row + 0.5) / ROWS) * 2;
  sweepDelay[i] = (col / (COLS - 1)) * SWEEP_MS;
}

const FlipdotWebGL = ({ discs }) => {
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const flipRef = useRef(null);
  const colorRef = useRef(null);
  const updatedAtRef = useRef(new Float64Array(COUNT));
  // Redraw only while something is flipping or after a resize; otherwise the
  // canvas already shows the right picture and a still scene costs nothing.
  const animateUntilRef = useRef(0);
  const dirtyRef = useRef(true);
  // Bumped to build the renderer again: after the GPU was reset under it (a
  // driver update, sleep), or when WebGL wasn't available yet.
  const [glEpoch, setGlEpoch] = useState(0);

  if (flipRef.current === null) {
    flipRef.current = new Float32Array(COUNT);
    randomFlips(flipRef.current);
    colorRef.current = new Float32Array(COUNT * 3);
    for (let i = 0; i < COUNT; i++) colorRef.current.set(randomLeafColor(), i * 3);
  }

  // New discs from the backend: only the ones that changed flip over (and get a
  // fresh colour if they turn on); the rest are left alone. No discs means no
  // connection -> noise.
  useEffect(() => {
    const flips = flipRef.current;
    if (!discs) {
      randomFlips(flips);
      dirtyRef.current = true;
      return;
    }
    const now = performance.now();
    const updatedAt = updatedAtRef.current;
    let changed = false;
    for (let i = 0; i < COUNT; i++) {
      if (discs[i] === flips[i]) continue;
      flips[i] = discs[i];
      // Sent back before it got over: turn back from where it is (the same
      // angle, mirrored) instead of snapping to its starting face. The turn
      // goes as u^2, so the mirrored point is sqrt(1 - u^2).
      const u = (now - updatedAt[i] - sweepDelay[i]) / TURN_MS;
      updatedAt[i] = u > 0 && u < 1 ? now - sweepDelay[i] - Math.sqrt(1 - u * u) * TURN_MS : now;
      if (discs[i] === 1) colorRef.current.set(randomLeafColor(), i * 3);
      changed = true;
    }
    if (changed) animateUntilRef.current = now + SWEEP_MS + FLIP_MS;
  }, [discs]);

  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    let regl;
    try {
      regl = createREGL({ canvas });
    } catch (error) {
      console.error("Error initializing WebGL, retrying:", error);
      const retry = setTimeout(() => setGlEpoch((n) => n + 1), 2000);
      return () => clearTimeout(retry);
    }
    // regl asks for a lost context back and restores its buffers, but the
    // restored canvas is blank and only redrawn when discs change -- on a
    // still scene, never. Starting over is the dependable way back.
    regl.on("restore", () => setGlEpoch((n) => n + 1));
    // regl's frame loop keeps running while the context is lost (its stopRAF
    // doesn't cancel), and drawing then throws "(regl) context lost".
    let lost = false;
    regl.on("lost", () => {
      lost = true;
    });

    // Letterbox the grid and size the discs in device pixels. Recomputed on
    // every container resize, so the dots never stretch or go stale.
    const view = { scale: [1, 1], pointSize: 1 };
    const maxPointSize = regl.limits.pointSizeDims[1];
    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(container.clientWidth * dpr));
      canvas.height = Math.max(1, Math.round(container.clientHeight * dpr));
      const pitch = Math.min(canvas.width / COLS, canvas.height / ROWS);
      view.scale = [(pitch * COLS) / canvas.width, (pitch * ROWS) / canvas.height];
      view.pointSize = Math.min(pitch * DOT_SIZE, maxPointSize);
      dirtyRef.current = true; // resizing clears the canvas
    };
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    resize();

    const progress = new Float32Array(COUNT);
    const flipBuffer = regl.buffer({ usage: "dynamic", data: flipRef.current });
    const progressBuffer = regl.buffer({ usage: "dynamic", data: progress });
    const colorBuffer = regl.buffer({ usage: "dynamic", data: colorRef.current });

    const drawDots = regl({
      vert: `
        precision mediump float;
        attribute vec2 position;
        attribute float flip;
        attribute float animationProgress;
        attribute vec3 dotColor;
        uniform vec2 scale;
        uniform float pointSize;
        varying float vFlip;
        varying float vAnimationProgress;
        varying vec3 vDotColor;

        void main() {
          vFlip = flip;
          vAnimationProgress = animationProgress;
          vDotColor = dotColor;
          gl_Position = vec4(position * scale, 0, 1);
          gl_PointSize = pointSize;
        }
      `,
      frag: `
        precision mediump float;
        varying float vFlip;
        varying float vAnimationProgress;
        varying vec3 vDotColor;

        const float PI = 3.14159265;
        const float TURN = ${TURN_SHARE.toFixed(4)};
        const float BOUNCE = 0.8;       // radians the disc springs back off its stop
        const float EDGE = 0.08;        // its thickness, seen edge-on
        const float PERSPECTIVE = 0.35;
        const vec3 LIGHT = vec3(0.0, 0.6, 0.8); // from above and in front

        // How far the disc has turned over, 0..PI.
        float flipAngle(float t) {
          if (t < TURN) {
            float u = t / TURN;
            return PI * u * u;  // pulled over faster and faster
          }
          float u = (t - TURN) / (1.0 - TURN);
          return PI - BOUNCE * abs(sin(2.0 * PI * u)) * (1.0 - u);  // two bounces, dying away
        }

        void main() {
          // The disc turns over about its horizontal axis: at 0 the face it is
          // leaving is up, at PI the new one. Seen from the front it flattens
          // to an edge and opens out again.
          float angle = flipAngle(vAnimationProgress);
          float c = cos(angle);
          float s = sin(angle);
          vec2 centered = gl_PointCoord - vec2(0.5);
          centered.y /= max(abs(c), EDGE);
          // The half tipping towards the viewer is nearer, so wider.
          centered.x *= 1.0 + PERSPECTIVE * s * centered.y;
          if (length(centered) > 0.5) discard;
          vec2 face = centered + vec2(0.5); // the pattern squashes with the disc

          // Until it is edge-on, the disc still shows the face it is leaving.
          float lit = angle < PI * 0.5 ? 1.0 - vFlip : vFlip;

          // Real discs have a round notch in the edge for the coil core. It sits
          // at the top of the coloured face, so turned over it is at the bottom.
          vec2 notch = vec2(0.5, lit > 0.5 ? 0.0 : 1.0);
          if (distance(face, notch) < 0.14) discard;

          float veinPattern = 0.0;
          float mainVein = smoothstep(0.05, 0.0, abs(face.x - 0.5));
          for (int i = 1; i <= 3; i++) {
            float y = float(i) * 0.2;
            float sideVein = smoothstep(0.03, 0.0, abs(face.y - y)) *
                             smoothstep(0.0, 0.5, face.x);
            veinPattern += sideVein * 0.3;
          }
          veinPattern += mainVein * 0.5;

          vec3 offColor = vec3(0.1, 0.1, 0.1);
          vec3 onColor = vDotColor;
          onColor = onColor * (1.0 - veinPattern * 0.3);

          vec3 color = mix(offColor, onColor, lit);
          // Edge-on, what shows is the disc's rim.
          color = mix(color, vec3(0.35), smoothstep(2.0 * EDGE, EDGE, abs(c)));

          // Lit from above: tipping up catches more light, tipping down less.
          // Face-on it sits at 0.7, as before.
          vec3 normal = vec3(0.0, -s, c) * sign(c);
          float light = max(dot(normal, normalize(LIGHT)), 0.0) / normalize(LIGHT).z;
          float brightness = 0.7 * (0.45 + 0.55 * light);

          gl_FragColor = vec4(color * brightness, 1.0);
        }
      `,
      attributes: {
        position: positions,
        flip: flipBuffer,
        animationProgress: progressBuffer,
        dotColor: colorBuffer,
      },
      uniforms: {
        scale: () => view.scale,
        pointSize: () => view.pointSize,
      },
      count: COUNT,
      primitive: "points",
    });

    let lastDraw = 0;
    const loop = regl.frame(() => {
      const now = performance.now();
      if (lost) return; // the restore rebuilds everything anyway
      // Nothing flipping since the last draw, nothing resized: skip the frame.
      if (!dirtyRef.current && lastDraw > animateUntilRef.current) return;
      dirtyRef.current = false;
      lastDraw = now;

      const updatedAt = updatedAtRef.current;
      for (let i = 0; i < COUNT; i++) {
        const t = (now - updatedAt[i] - sweepDelay[i]) / FLIP_MS;
        progress[i] = t < 0 ? 0 : t > 1 ? 1 : t;
      }
      flipBuffer.subdata(flipRef.current);
      progressBuffer.subdata(progress);
      colorBuffer.subdata(colorRef.current);

      regl.clear({ color: [0.1, 0.1, 0.1, 1], depth: 1 });
      drawDots();
    });

    return () => {
      observer.disconnect();
      loop.cancel();
      regl.destroy();
    };
  }, [glEpoch]);

  return (
    <div ref={containerRef} className="h-full w-full">
      <canvas ref={canvasRef} className="block h-full w-full" />
    </div>
  );
};

export default FlipdotWebGL;
