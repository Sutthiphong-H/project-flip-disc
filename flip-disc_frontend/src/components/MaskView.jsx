import { useEffect, useRef } from "react";
import { COLS, ROWS } from "./Flipdot";

// The raw 80x45 matrix, one pixel per disc, scaled up without smoothing.
const MaskView = ({ matrix }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const ctx = canvasRef.current.getContext("2d");
    const image = ctx.createImageData(COLS, ROWS);
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const v = matrix?.[r]?.[c] ? 255 : 0;
        const i = (r * COLS + c) * 4;
        image.data[i] = image.data[i + 1] = image.data[i + 2] = v;
        image.data[i + 3] = 255;
      }
    }
    ctx.putImageData(image, 0, 0);
  }, [matrix]);

  return (
    <canvas
      ref={canvasRef}
      width={COLS}
      height={ROWS}
      className="block h-full w-full object-contain [image-rendering:pixelated]"
    />
  );
};

export default MaskView;
