# FlipDisc Project

FlipDisc Project was made for education of Project.
Chadsadakorn Piriyasiriphan
Jadesalit Suriyanimitsuk
Sutthiphong Hongkham

A webcam feed is segmented with RVM (Robust Video Matting; U2NET optional), each silhouette's distance is estimated from
its height, and everyone beyond `FLIPDISC_MAX_DISTANCE` is dropped. What's left is
reduced to an 80x45 black-and-white matrix and streamed to the React frontend over
Socket.IO. With nobody in range the backend loops an idle video instead.

## Layout

```
backend/
├── assets/            idle video clips
├── models/            model weights (not committed)
├── requirements.txt
├── src/
│   ├── app.py         entry point
│   ├── settings.py    every tunable, overridable by environment variable
│   ├── pipeline.py    capture -> segment -> publish loop
│   ├── frame_rate.py  rolling FPS meter
│   ├── sources/       webcam capture, idle video playback
│   ├── vision/        RVM + U2NET models, segmentation worker
│   └── server/        Flask + Socket.IO server and publisher
└── tools/camtest.py   preview a camera to find its index
flip-disc_frontend/    React client
arduino/               ESP32 sketches
```

## Installation

The RTX 50-series needs a CUDA 12.8 build of torch, so install torch first:

```bash
cd backend
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Put the weights in `backend/models/`:

- `rvm_mobilenetv3.pth` -- the default model ([download](https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth))
- `u2net_human_seg.pth` -- only needed for `FLIPDISC_MODEL=u2net` ([download](https://drive.usercontent.google.com/download?id=1m_Kgs91b21gayc2XLW0ou8yugAIadWVP&export=download&authuser=0&confirm=t))

Put the idle clip at `backend/assets/video.mp4` (not in git -- it's 90 MB).

RVM's code in `backend/src/vision/rvm/` is vendored from [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) and is GPL-3.0.

```bash
cd flip-disc_frontend
npm install
```

## Running

Build the frontend once and the backend serves it -- open http://localhost:5000:

```bash
cd flip-disc_frontend
npm run build            # -> flip-disc_frontend/dist (FLIPDISC_FRONTEND_DIST to move it)
cd ../backend
python src/app.py        # Flask + Socket.IO + the built page, all on port 5000
```

While working on the frontend, use Vite's dev server instead (http://localhost:5173,
talks to the backend on port 5000):

```bash
cd flip-disc_frontend
npm run dev
```

### Running unattended

`backend/run-backend.bat` runs the backend and restarts it whenever it exits with an
error. The backend exits with code 1 by itself when something breaks and won't
recover in place: the pipeline thread dies (a model that won't load, a CUDA context
broken by a GPU driver reset), segmentation hangs, or no frame goes out for
`FLIPDISC_STALL_SECONDS`. Frames go out in every mode (the idle clip plays even
without a camera), so a gap in them always means the backend is stuck. Ctrl+C exits with 0 and ends
the script too.

To start it with Windows, create a Task Scheduler task with the trigger **At log on**
and the action `backend\run-backend.bat`. Use log on rather than a Windows service:
services run outside the desktop session, and webcam access from there is unreliable. On the display machine, also turn off
sleep and USB selective suspend (Power Options → USB settings), or the webcam
can be powered down.

- `GET /status` returns 503 with `"status": "stalled"` once frames stop, for an
  external health check. Before the first frame the status is `"starting"`.
- The page shows *No frames from the backend* when it is connected but nothing has
  arrived for 3 s, and *Disconnected* once the backend has exited.
- If WebGL loses its context (a GPU reset, a driver update), the flip-disc view
  rebuilds itself. You don't need to reload.

Cameras are managed from the camera icon at the top right of the page: pick the
camera in use from the dropdown, and add, edit or delete cameras.

- **Webcam**: *Scan* shows the webcams plugged into the backend machine (with a
  thumbnail each); pick one, give it a name and a resolution.
- **IP camera**: name, IP, port, username, password, channel, main/sub stream,
  brand and resolution. The RTSP URL is built from those: Dahua
  `.../cam/realmonitor?channel=1&subtype=0` or Hikvision `.../Streaming/Channels/101`.
- *Test* grabs one frame and shows it. The backend only switches to a camera that
  delivers a picture, otherwise the current one keeps running.

The list lives in `backend/cameras.json` -- not committed, since it holds the IP
cameras' passwords, which the page never gets back (a blank password when editing
keeps the saved one). Without that file the backend starts with webcam `FLIPDISC_CAMERA`.

The page has three views: **Flip-disc** (the simulated display), **Mask** (the raw
80x45 matrix) and **Camera** (the webcam with the silhouette highlighted and a
distance box per person, streamed from `/debug.mjpg` only while it's open).

## Environment variables

| Variable                   | Default      | Meaning                                  |
| -------------------------- | ------------ | ---------------------------------------- |
| `FLIPDISC_CAMERA`          | `0`          | First webcam, until `cameras.json` exists |
| `FLIPDISC_PORT`            | `5000`       | Socket.IO / HTTP port                    |
| `FLIPDISC_EMIT_FPS`        | `30`         | Upper bound on frames sent to the client |
| `FLIPDISC_MAX_DISTANCE`    | `15`         | People further than this (m) are dropped |
| `FLIPDISC_FOCAL_LENGTH`    | `460`        | For the distance estimate; calibrate it  |
| `FLIPDISC_MIN_AREA`        | `0.005`      | Blobs under this share of frame = noise  |
| `FLIPDISC_MODEL`           | `rvm`        | `rvm` or `u2net`                         |
| `FLIPDISC_PROCESS_SIZE`    | `640x360`    | Model input size, 16:9 (u2net: 512x288)  |
| `FLIPDISC_SMOOTHING`       | `0.5`        | Anti-flicker; 1 = off, lower = steadier  |
| `FLIPDISC_IDLE_TIMEOUT`    | `15`         | Seconds without a person before idle     |
| `FLIPDISC_IDLE_VIDEO`      | `video.mp4`  | Clip in `backend/assets/` to loop        |
| `FLIPDISC_STALL_SECONDS`   | `10`         | No frame this long: exit for a restart   |
| `FLIPDISC_STARTUP_SECONDS` | `120`        | No first frame this long: same           |
