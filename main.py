import sys
import os
import warnings

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QLineEdit, QMessageBox
)
from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt, QSize, QCoreApplication

import sounddevice as sd
from scipy.io.wavfile import write

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pickle

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import requests
import subprocess

# ────────── 설정 ──────────
warnings.filterwarnings("ignore", message=".*weights_only=False.*")
QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

# ────────── 1) 모델 로딩 ──────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
num_labels = len(label_encoder.classes_)

bert_model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-multilingual-cased", num_labels=num_labels
).to(DEVICE)
bert_model.load_state_dict(torch.load("bert_model.pt", map_location=DEVICE, weights_only=True))
bert_model.eval()
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")

class CNNClassifier(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.backbone = models.resnet50(pretrained=False)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Identity(),
            nn.Linear(in_feats, num_labels)
        )
    def forward(self, x):
        return self.backbone(x)

cnn_model = CNNClassifier(num_labels=num_labels).to(DEVICE)
cnn_model.load_state_dict(torch.load("cnn_model_res50.pt", map_location=DEVICE, weights_only=True))
cnn_model.eval()

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ────────── 2) 예측 & 외부 API 함수 ──────────
def predict_ost_from_text(text: str) -> str:
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64).to(DEVICE)
    with torch.no_grad():
        out = bert_model(**enc)
        idx = out.logits.argmax(dim=1).item()
    return label_encoder.inverse_transform([idx])[0]

def predict_ost_from_image(path: str) -> str:
    img = Image.open(path).convert("RGB")
    t = image_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = cnn_model(t)
        idx = out.argmax(dim=1).item()
    return label_encoder.inverse_transform([idx])[0]

AUDD_API_KEY = "c2acb8c40d20a8916566c4e5997ab2cc"
SPOTIFY_CLIENT_ID = "04df9d7a817d4709a27eee2e1ecfb2f2"
SPOTIFY_CLIENT_SECRET = "d36b326fc5df4a97b3ba1a96f13280a2"

def record_audio_wav(duration=10, output_file="temp_audio.wav") -> str:
    QMessageBox.information(None, "녹음 안내", f"{duration}초간 음악을 들려주세요.")
    fs = 44100
    rec = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(output_file, fs, rec)
    return output_file

def convert_wav_to_mp3(wav, mp3):
    subprocess.run(["ffmpeg", "-y", "-i", wav, mp3],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def recognize_with_audd(mp3: str) -> dict:
    url = "https://api.audd.io/"
    with open(mp3, "rb") as f:
        files = {"file": f}
        data = {"api_token": AUDD_API_KEY, "return": "spotify"}
        r = requests.post(url, data=data, files=files)
    if r.status_code == 200:
        return r.json()
    return {}

def search_spotify_for_cover(artist: str, title: str) -> dict:
    auth = SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET
    )
    sp = spotipy.Spotify(auth_manager=auth)

    for q in [f"{title} {artist}", f"{artist} {title}"]:
        res = sp.search(q=q, limit=1, type="track")
        items = res.get("tracks", {}).get("items", [])
        if items:
            t = items[0]
            return {
                "title":  t["name"],
                "artist": t["artists"][0]["name"],
                "cover_url": t["album"]["images"][0]["url"]
            }
    return {}

# ────────── 3) PyQt5 GUI 정의 ──────────
class DeepTuneApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepTune 시뮬레이터")
        self.setFixedSize(375, 812)
        self._setup_ui()

    def _setup_ui(self):
        # 배경
        self.bg = QLabel(self)
        if os.path.exists("assets/background.png"):
            pix = QPixmap("assets/background.png")
            self.bg.setPixmap(pix)
            self.bg.setGeometry(0, 0, 375, 812)
            self.bg.lower()
        else:
            self.bg.setStyleSheet("background-color:#1D2671;")

        # 헤더
        self.header = QLabel("음악을 찾아보세요", self)
        self.header.setFont(QFont("Helvetica", 16, QFont.Bold))
        self.header.setStyleSheet("color:white;")
        self.header.setAlignment(Qt.AlignCenter)
        self.header.setGeometry(0, 30, 375, 40)

        # 마이크 버튼
        self.mic_btn = QPushButton(self)
        self.mic_btn.setIcon(QIcon("assets/mic2.png"))
        self.mic_btn.setIconSize(QSize(30, 30))
        self.mic_btn.setGeometry(20, 35, 30, 30)
        self.mic_btn.setStyleSheet("border:none;")
        self.mic_btn.clicked.connect(self.on_click_audio)
        self.mic_label = QLabel("음성", self)
        self.mic_label.setFont(QFont("Helvetica", 10))
        self.mic_label.setStyleSheet("color:white;")
        self.mic_label.setAlignment(Qt.AlignCenter)
        self.mic_label.setGeometry(10, 70, 50, 20)

        # 이미지 버튼
        self.img_btn = QPushButton(self)
        self.img_btn.setIcon(QIcon("assets/photo.png"))
        self.img_btn.setIconSize(QSize(30, 30))
        self.img_btn.setGeometry(325, 35, 30, 30)
        self.img_btn.setStyleSheet("border:none;")
        self.img_btn.clicked.connect(self.on_click_image)
        self.img_label = QLabel("사진", self)
        self.img_label.setFont(QFont("Helvetica", 10))
        self.img_label.setStyleSheet("color:white;")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setGeometry(315, 70, 50, 20)

        # 앨범 커버
        self.album = QLabel(self)
        self.album.setGeometry(107, 256, 160, 160)
        self.album.setStyleSheet("background-color:rgba(255,255,255,30);border-radius:12px;")

        # 곡 제목 / 아티스트
        self.song_title = QLabel("노래 제목", self)
        self.song_title.setFont(QFont("Helvetica", 14, QFont.Bold))
        self.song_title.setStyleSheet("color:white;")
        self.song_title.setAlignment(Qt.AlignCenter)
        self.song_title.setGeometry(0, 430, 375, 30)
        self.artist = QLabel("아티스트", self)
        self.artist.setFont(QFont("Helvetica", 11))
        self.artist.setStyleSheet("color:#DADADA;")
        self.artist.setAlignment(Qt.AlignCenter)
        self.artist.setGeometry(0, 460, 375, 25)

        # 텍스트 입력창
        self.input = QLineEdit(self)
        self.input.setPlaceholderText("장면 설명을 입력하세요")
        self.input.setStyleSheet(
            "padding:8px;font-size:13px;border-radius:20px;"
            "background-color:rgba(255,255,255,180);"
        )
        self.input.setGeometry(60, 740, 255, 50)
        self.input.returnPressed.connect(self.on_click_text)

    def _show_result(self, title, artist, cover_url):
        # 제목/아티스트 업데이트
        self.song_title.setText(title or "곡 정보 없음")
        self.artist.setText(artist or "아티스트 없음")

        # 기본 커버 셋업
        pix = QPixmap("assets/default_cover.jpg") if os.path.exists("assets/default_cover.jpg") else QPixmap()

        # Spotify cover URL이 있으면 동기 다운로드
        if cover_url and cover_url.startswith("http"):
            try:
                resp = requests.get(cover_url, timeout=5)
                resp.raise_for_status()
                tmp = QPixmap()
                if tmp.loadFromData(resp.content):
                    pix = tmp
            except Exception as e:
                print(f"Cover download failed: {e}")

        # QLabel에 설정
        self.album.setPixmap(pix.scaled(160, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def on_click_text(self):
        txt = self.input.text().strip()
        if not txt:
            QMessageBox.warning(self, "입력 오류", "장면 설명을 입력해주세요.")
            return
        try:
            label = predict_ost_from_text(txt)
            artist, title = label.split(" - ", 1)
            info = search_spotify_for_cover(artist, title)
            self._show_result(
                title     = info.get("title", title),
                artist    = info.get("artist", artist),
                cover_url = info.get("cover_url", "")
            )
        except Exception as e:
            QMessageBox.warning(self, "예측 오류", str(e))

    def on_click_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "파일 선택", "", "Images (*.png *.jpg)")
        if not path:
            return
        try:
            label = predict_ost_from_image(path)
            artist, title = label.split(" - ", 1)
            info = search_spotify_for_cover(artist, title)
            self._show_result(
                title     = info.get("title", title),
                artist    = info.get("artist", artist),
                cover_url = info.get("cover_url", "")
            )
        except Exception as e:
            QMessageBox.warning(self, "예측 오류", str(e))

    def on_click_audio(self):
        try:
            wav = record_audio_wav(duration=5, output_file="temp.wav")
            mp3 = "temp.mp3"
            convert_wav_to_mp3(wav, mp3)
            result = recognize_with_audd(mp3)
            if result and "result" in result:
                audd = result["result"]
                title, artist = audd.get("title",""), audd.get("artist","")
                info = search_spotify_for_cover(artist, title)
                self._show_result(
                    title     = info.get("title", title),
                    artist    = info.get("artist", artist),
                    cover_url = info.get("cover_url", "")
                )
            else:
                self._show_result("인식 실패", "", "")
        except Exception as e:
            QMessageBox.warning(self, "오류", str(e))
        finally:
            for fn in ("temp.wav", "temp.mp3"):
                if os.path.exists(fn):
                    os.remove(fn)

# ────────── 4) 앱 실행 ──────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DeepTuneApp()
    window.show()
    sys.exit(app.exec_())
