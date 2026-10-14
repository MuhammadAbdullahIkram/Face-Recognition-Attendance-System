"""Core logic: face detection + recognition + attendance (OpenCV YuNet + SFace).

Used by the Colab notebook, the Streamlit app (app.py) and realtime.py.
No dlib / heavy install needed - only OpenCV (>= 4.5.4), NumPy and Pandas.
"""
import pickle
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import pandas as pd

TIMEZONE = "Asia/Karachi"          # attendance timestamps (Colab runs in UTC)
MODEL_DIR = Path("models")
DATA_DIR = Path("data")
EMB_FILE = DATA_DIR / "embeddings.pkl"
ATT_FILE = DATA_DIR / "attendance.csv"
COSINE_THRESHOLD = 0.363           # recommended SFace cosine threshold

_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = {
    "face_detection_yunet_2023mar.onnx": f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def download_models():
    """Download the two ONNX models once (YuNet ~0.2 MB, SFace ~37 MB)."""
    MODEL_DIR.mkdir(exist_ok=True)
    for name, url in MODELS.items():
        path = MODEL_DIR / name
        if not path.exists() or path.stat().st_size < 100_000:
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, path)
        if path.stat().st_size < 100_000:
            raise RuntimeError(f"{name} download looks broken - check your internet connection.")


def _now():
    return datetime.now(ZoneInfo(TIMEZONE))


class FaceEngine:
    def __init__(self, score_threshold=0.8, match_threshold=COSINE_THRESHOLD):
        download_models()
        self.detector = cv2.FaceDetectorYN.create(
            str(MODEL_DIR / "face_detection_yunet_2023mar.onnx"), "", (320, 320),
            score_threshold, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(
            str(MODEL_DIR / "face_recognition_sface_2021dec.onnx"), "")
        self.threshold = match_threshold
        self.db = {}                # name -> list of 128-d embeddings
        self.load()

    # ---------- storage ----------
    def load(self):
        if EMB_FILE.exists():
            with open(EMB_FILE, "rb") as f:
                self.db = pickle.load(f)

    def save(self):
        DATA_DIR.mkdir(exist_ok=True)
        with open(EMB_FILE, "wb") as f:
            pickle.dump(self.db, f)

    def people(self):
        return {name: len(embs) for name, embs in self.db.items()}

    def delete(self, name):
        self.db.pop(name, None)
        self.save()

    # ---------- detection / embedding ----------
    def detect(self, bgr):
        h, w = bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(bgr)
        return [] if faces is None else list(faces)

    def embed(self, bgr, face):
        aligned = self.recognizer.alignCrop(bgr, face)
        return self.recognizer.feature(aligned)

    # ---------- registration ----------
    def register(self, name, bgr, save=True):
        """Add the largest face in the image to `name`. Returns True on success."""
        faces = self.detect(bgr)
        if not faces:
            return False
        face = max(faces, key=lambda f: f[2] * f[3])
        self.db.setdefault(name, []).append(self.embed(bgr, face))
        if save:
            self.save()
        return True

    def register_folder(self, root="dataset"):
        """dataset/<Person Name>/*.jpg  ->  {name: images_registered}"""
        counts = {}
        for person in sorted(Path(root).iterdir()):
            if not person.is_dir():
                continue
            ok = 0
            for img_path in person.iterdir():
                if img_path.suffix.lower() in IMG_EXT:
                    img = cv2.imread(str(img_path))
                    if img is not None and self.register(person.name, img, save=False):
                        ok += 1
            counts[person.name] = ok
        self.save()
        return counts

    # ---------- recognition ----------
    def identify(self, emb):
        best_name, best_score = "Unknown", -1.0
        for name, embs in self.db.items():
            for e in embs:
                s = self.recognizer.match(emb, e, cv2.FaceRecognizerSF_FR_COSINE)
                if s > best_score:
                    best_name, best_score = name, s
        if best_score < self.threshold:
            return "Unknown", best_score
        return best_name, best_score

    def recognize(self, bgr):
        results = []
        for f in self.detect(bgr):
            name, score = self.identify(self.embed(bgr, f))
            x, y, w, h = f[:4].astype(int)
            results.append({"name": name, "score": float(score), "box": (x, y, w, h)})
        return results

    @staticmethod
    def annotate(bgr, results):
        out = bgr.copy()
        for r in results:
            x, y, w, h = r["box"]
            color = (0, 200, 0) if r["name"] != "Unknown" else (0, 0, 255)
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            cv2.putText(out, f'{r["name"]} ({r["score"]:.2f})', (x, max(y - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return out


# ---------- attendance ----------
def load_attendance(path=ATT_FILE):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path, dtype=str)
    return pd.DataFrame(columns=["Name", "Date", "Time"])


def mark_attendance(name, path=ATT_FILE):
    """One entry per person per day. Returns True if newly marked."""
    path = Path(path)
    now = _now()
    date, time = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
    df = load_attendance(path)
    if ((df["Name"] == name) & (df["Date"] == date)).any():
        return False
    df.loc[len(df)] = [name, date, time]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return True
