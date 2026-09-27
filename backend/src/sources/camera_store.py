"""The saved camera list, edited from the web UI and kept in cameras.json.

    {"active": 1,
     "cameras": [
       {"id": 1, "name": "Webcam", "type": "webcam", "index": 0, "resolution": [1280, 720]},
       {"id": 2, "name": "Front door", "type": "ip", "brand": "dahua",
        "ip": "192.168.1.188", "port": 554, "username": "admin", "password": "...",
        "channel": 1, "stream": "main", "resolution": [1280, 720]}]}

Modelled on Aicam-detection-Pi5's CameraSpec: a camera's hard signature
(type, source, resolution) is what needs the capture restarted; renaming it
doesn't. Passwords stay in this file: public() is what a browser gets, and
the RTSP URL is only ever printed through redact().
"""

import json
import re
import threading
from urllib.parse import quote

from settings import CAMERA_INDEX, CAMERAS_PATH, INPUT_RESOLUTION

# 16:9 like the display; anything else would be cropped anyway.
RESOLUTIONS = [(640, 360), (960, 540), (1280, 720), (1920, 1080)]
BRANDS = ("dahua", "hikvision")
STREAMS = ("main", "sub")

_HOST = re.compile(r"^[A-Za-z0-9.\-]{1,253}$")  # an IP address or a host name, nothing else
_USERINFO = re.compile(r"(?<=://)[^/@\s]+@")


def redact(text):
    """rtsp://user:pass@host/... -> rtsp://***@host/..."""
    return _USERINFO.sub("***@", text)


def camera_url(cam):
    """The RTSP URL for an IP camera entry.

    Dahua (and the many OEMs using its firmware):
        rtsp://user:pass@ip:554/cam/realmonitor?channel=1&subtype=0   (0 main, 1 sub)
    Hikvision:
        rtsp://user:pass@ip:554/Streaming/Channels/101                (channel 1: 101 main, 102 sub)
    """
    login = ""
    if cam["username"]:
        login = quote(cam["username"], safe="")
        if cam["password"]:
            login += ":" + quote(cam["password"], safe="")
        login += "@"
    base = f"rtsp://{login}{cam['ip']}:{cam['port']}"
    if cam["brand"] == "hikvision":
        return f"{base}/Streaming/Channels/{cam['channel']}0{1 if cam['stream'] == 'main' else 2}"
    return f"{base}/cam/realmonitor?channel={cam['channel']}&subtype={0 if cam['stream'] == 'main' else 1}"


def source(cam):
    """What gets opened: a webcam index or an RTSP URL."""
    return cam["index"] if cam["type"] == "webcam" else camera_url(cam)


def hard_signature(cam):
    """The fields whose change needs the capture restarted."""
    return (cam["type"], source(cam), tuple(cam["resolution"]))


def summary(cam):
    width, height = cam["resolution"]
    if cam["type"] == "webcam":
        return f"Webcam {cam['index']} · {width}×{height}"
    return f"{cam['brand'].title()} {cam['ip']} · ch {cam['channel']} {cam['stream']} · {width}×{height}"


def public(cam):
    """A camera as a browser may see it: no password, URL masked."""
    out = {k: v for k, v in cam.items() if k != "password"}
    out["summary"] = summary(cam)
    if cam["type"] == "ip":
        out["has_password"] = bool(cam["password"])
        out["url"] = redact(camera_url(cam))
    return out


def _int(data, key, low, high, default=None):
    value = data.get(key, default)
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number") from None
    if not low <= value <= high:
        raise ValueError(f"{key} must be between {low} and {high}")
    return value


def validate(data, existing=None):
    """Return a clean camera entry built from form data, or raise ValueError.

    `existing` is the entry being edited: its id is kept, and so is its
    password when the form leaves the password blank (browsers never get it).
    """
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    name = str(data.get("name", "")).strip()
    if not name or len(name) > 40:
        raise ValueError("give the camera a name (up to 40 characters)")
    try:
        resolution = tuple(int(v) for v in data.get("resolution", ()))
    except (TypeError, ValueError):
        resolution = ()
    if resolution not in RESOLUTIONS:
        raise ValueError("pick one of the listed resolutions")

    cam = {"id": existing["id"] if existing else None, "name": name,
           "type": data.get("type"), "resolution": list(resolution)}
    if cam["type"] == "webcam":
        cam["index"] = _int(data, "index", 0, 99)
        return cam
    if cam["type"] != "ip":
        raise ValueError("type must be webcam or ip")

    ip = str(data.get("ip", "")).strip()
    if not _HOST.match(ip):
        raise ValueError("IP must be an address like 192.168.1.188 (no rtsp://, no port)")
    brand = data.get("brand", "dahua")
    if brand not in BRANDS:
        raise ValueError(f"brand must be one of {', '.join(BRANDS)}")
    stream = data.get("stream", "main")
    if stream not in STREAMS:
        raise ValueError("stream must be main or sub")
    password = str(data.get("password") or "")
    if not password and existing and existing.get("type") == "ip":
        password = existing["password"]
    cam.update(brand=brand, ip=ip, port=_int(data, "port", 1, 65535, 554),
               username=str(data.get("username") or "").strip(), password=password,
               channel=_int(data, "channel", 1, 999, 1), stream=stream)
    return cam


class CameraStore:
    """The camera list plus which one is active; every change is saved at once."""

    def __init__(self, path=CAMERAS_PATH):
        self.path = path
        self._lock = threading.Lock()
        self.active_id, self.cameras = self._load()

    def _load(self):
        default = [{"id": 1, "name": "Webcam", "type": "webcam", "index": CAMERA_INDEX,
                    "resolution": list(INPUT_RESOLUTION)}]
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            cameras = []
            for raw in data["cameras"]:
                cam = validate(raw, existing={"id": int(raw["id"]), **raw})
                cameras.append(cam)
            active = int(data["active"])
            if cameras and any(c["id"] == active for c in cameras):
                return active, cameras
            raise ValueError("the active camera is not in the list")
        except FileNotFoundError:
            return 1, default
        except (OSError, ValueError, KeyError, TypeError) as e:
            print(f"Ignoring {self.path.name} ({e}); starting with webcam {CAMERA_INDEX}")
            return 1, default

    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"active": self.active_id, "cameras": self.cameras}, indent=2),
                       encoding="utf-8")
        tmp.replace(self.path)  # never leave a half-written file behind

    def get(self, cam_id):
        with self._lock:
            return next((dict(c) for c in self.cameras if c["id"] == cam_id), None)

    @property
    def active(self):
        return self.get(self.active_id)

    def listing(self):
        with self._lock:
            return {"active": self.active_id, "cameras": [public(c) for c in self.cameras]}

    def add(self, cam):
        with self._lock:
            cam = dict(cam, id=max((c["id"] for c in self.cameras), default=0) + 1)
            self.cameras.append(cam)
            self._save()
            return cam

    def update(self, cam):
        with self._lock:
            self.cameras = [cam if c["id"] == cam["id"] else c for c in self.cameras]
            self._save()

    def remove(self, cam_id):
        with self._lock:
            if cam_id == self.active_id:
                raise ValueError("switch to another camera before deleting this one")
            self.cameras = [c for c in self.cameras if c["id"] != cam_id]
            self._save()

    def set_active(self, cam_id):
        with self._lock:
            self.active_id = cam_id
            self._save()
