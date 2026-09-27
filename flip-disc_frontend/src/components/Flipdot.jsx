import { useEffect, useRef } from "react";
import createREGL from "regl";

export const ROWS = 45;
export const COLS = 80;
const COUNT = ROWS * COLS;
const ANIMATION_DURATION = 100; // Animation duration in ms
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
for (let i = 0; i < COUNT; i++) {
  const row = Math.floor(i / COLS);
  const col = i % COLS;
  positions[i * 2] = ((col + 0.5) / COLS) * 2 - 1;
  positions[i * 2 + 1] = 1 - ((row + 0.5) / ROWS) * 2;
}

const FlipdotWebGL = ({ matrix }) => {
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const flipRef = useRef(null);
  const colorRef = useRef(null);
  const updatedAtRef = useRef(new Float64Array(COUNT));

  if (flipRef.current === null) {
    flipRef.current = new Float32Array(COUNT);
    randomFlips(flipRef.current);
    colorRef.current = new Float32Array(COUNT * 3);
    for (let i = 0; i < COUNT; i++) colorRef.current.set(randomLeafColor(), i * 3);
  }

  // A new frame from the backend: flipped discs restart their animation, and
  // newly lit ones get a fresh colour. No frame means no connection -> noise.
  useEffect(() => {
    const flips = flipRef.current;
    if (!matrix) {
      randomFlips(flips);
      return;
    }
    const now = performance.now();
    for (let r = 0; r < Math.min(ROWS, matrix.length); r++) {
      const row = matrix[r];
      for (let c = 0; c < Math.min(COLS, row.length); c++) {
        const i = r * COLS + c;
        if (row[c] === flips[i]) continue;
        flips[i] = row[c];
        updatedAtRef.current[i] = now;
        if (row[c] === 1) colorRef.current.set(randomLeafColor(), i * 3);
      }
    }
  }, [matrix]);

  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    let regl;
    try {
      regl = createREGL({ canvas });
    } catch (error) {
      console.error("Error initializing WebGL:", error);
      return;
    }

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

        void main() {
          vec2 centered = gl_PointCoord - vec2(0.5);
          float dist = length(centered);

          if (dist > 0.5) discard;

          float t = vAnimationProgress;
          float animEffect = sin(t * 3.14);

          float veinPattern = 0.0;
          float mainVein = smoothstep(0.05, 0.0, abs(gl_PointCoord.x - 0.5));
          for (int i = 1; i <= 3; i++) {
            float y = float(i) * 0.2;
            float sideVein = smoothstep(0.03, 0.0, abs(gl_PointCoord.y - y)) *
                             smoothstep(0.0, 0.5, gl_PointCoord.x);
            veinPattern += sideVein * 0.3;
          }
          veinPattern += mainVein * 0.5;

          vec3 offColor = vec3(0.1, 0.1, 0.1);
          vec3 onColor = vDotColor;
          onColor = onColor * (1.0 - veinPattern * 0.3);

          vec3 color = mix(offColor, onColor, vFlip);
          float brightness = 0.7 + 0.3 * animEffect;

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

    const loop = regl.frame(() => {
      const now = performance.now();
      const updatedAt = updatedAtRef.current;
      for (let i = 0; i < COUNT; i++) {
        progress[i] = Math.min((now - updatedAt[i]) / ANIMATION_DURATION, 1.0);
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
  }, []);

  return (
    <div ref={containerRef} className="h-full w-full">
      <canvas ref={canvasRef} className="block h-full w-full" />
    </div>
  );
};

export default FlipdotWebGL;
