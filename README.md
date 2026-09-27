# FlipDisc Project

FlipDisc Project was made for education of Project.
Chadsadakorn Piriyasiriphan
Jadesalit Suriyanimitsuk
Sutthiphong Hongkham

A webcam feed is segmented (YOLO for person detection + U2NET for the silhouette),
reduced to an 80x45 black-and-white matrix and streamed to the React frontend over
Socket.IO. With nobody in front of the camera the backend loops an idle video instead.

## Layout

```
backend/
├── assets/            idle video clips
├── models/            model weights (not committed)
├── requirements.txt
├── src/
│   ├── app.py         entry point
│   ├── settings.py    every tunable, overridable by environment variable
│   ├── pipeline.py    capture -> detect -> segment -> publish loop
│   ├── frame_rate.py  rolling FPS meter
│   ├── sources/       webcam capture, idle video playback
│   ├── vision/        U2NET definition, segmentation worker, person detector
│   └── server/        Flask + Socket.IO server and publisher
└── tools/camtest.py   preview a camera to find its index
flip-disc_frontend/    React client
arduino/               ESP32 sketches
```

## Installation

The RTX 50-series needs a CUDA 12.8 build of torch, so install torch first:

```bash
cd backend
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Put `u2net_human_seg.pth` in `backend/models/`
([download](https://drive.usercontent.google.com/download?id=1m_Kgs91b21gayc2XLW0ou8yugAIadWVP&export=download&authuser=0&confirm=t)).
`yolo12n.pt` is fetched by ultralytics on first run.

```bash
cd flip-disc_frontend
npm install
```

## Running

```bash
cd backend
python src/app.py        # Flask + Socket.IO on port 5000
```

```bash
cd flip-disc_frontend
npm run dev
```

Not sure which camera index is the webcam? `python tools/camtest.py 1`

## Environment variables

| Variable                   | Default      | Meaning                                  |
| -------------------------- | ------------ | ---------------------------------------- |
| `FLIPDISC_CAMERA`          | `0`          | Webcam index                             |
| `FLIPDISC_DEBUG`           | `1`          | Show the OpenCV debug/preview windows    |
| `FLIPDISC_PORT`            | `5000`       | Socket.IO / HTTP port                    |
| `FLIPDISC_EMIT_FPS`        | `20`         | Upper bound on frames sent to the client |
| `FLIPDISC_DETECT_INTERVAL` | `15`         | Run YOLO every N frames                  |
| `FLIPDISC_CONFIDENCE`      | `0.75`       | Person detection confidence threshold    |
| `FLIPDISC_IDLE_TIMEOUT`    | `15`         | Seconds without a person before idle     |
| `FLIPDISC_IDLE_VIDEO`      | `video.mp4`  | Clip in `backend/assets/` to loop        |
