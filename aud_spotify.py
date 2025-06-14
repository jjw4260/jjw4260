# -*- coding: utf-8 -*-
# AudD + Spotify 음악 인식 통합 + 드라마 OST 예측 (BERT / CNN) 기능

import os
import sys
import json
import base64
import subprocess
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from sklearn.preprocessing import LabelEncoder
import pickle

import sounddevice as sd
import soundfile as sf

import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# (1) 경고 무시 설정
warnings.filterwarnings("ignore", message=".*weights_only=False.*")

# (2) API 키 / 토큰 설정
# Spotify API
SPOTIFY_CLIENT_ID     = "04df9d7a817d4709a27eee2e1ecfb2f2"
SPOTIFY_CLIENT_SECRET = "d36b326fc5df4a97b3ba1a96f13280a2"

# AudD API
AUDD_API_KEY = "757a1dc15f25bd48595392e44ca2acb6"

# (3) 녹음 → MP3 변환 함수 (AudD용)
def record_audio_mp3(filename="recorded.mp3", duration=8, samplerate=44100):
    """
    마이크에서 duration 초간 녹음 → 임시 WAV 저장 → ffmpeg 로 MP3로 변환 → 반환
    """
    temp_wav = "temp.wav"
    print("[⏺️] 녹음 시작...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    sf.write(temp_wav, recording, samplerate)
    print(f"[✔] 녹음 완료: {temp_wav}")

    subprocess.run(["ffmpeg", "-y", "-i", temp_wav, filename], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[✔] MP3 변환 완료: {filename}")
    return filename

# (4) AudD API 로 음악 인식
def recognize_with_audd(file_path):

    print(" AudD API로 음악 인식 중...")
    url = 'https://api.audd.io/'
    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {'api_token': AUDD_API_KEY, 'return': 'spotify'}
        response = requests.post(url, data=data, files=files)
    if response.status_code != 200:
        print(f" AudD API 실패: HTTP {response.status_code}")
        return None
    return response.json()

# (5) Spotify 검색 함수
def search_spotify(query):

    print(f" Spotify에서 '{query}' 검색 중...")
    auth_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID,
                                            client_secret=SPOTIFY_CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    result = sp.search(q=query, limit=1, type='track')
    if result['tracks']['items']:
        track = result['tracks']['items'][0]
        return {
            'title':   track['name'],
            'artist':  track['artists'][0]['name'],
            'album':   track['album']['name'],
            'preview': track['preview_url'],
            'cover':   track['album']['images'][0]['url']
        }
    else:
        return None

# (6) BERT 텍스트 기반 OST 예측 모델 로드
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# (6-1) 라벨 인코더 로드
label_encoder_path = "label_encoder.pkl"
if not os.path.exists(label_encoder_path):
    raise FileNotFoundError(f" 라벨 인코더 파일이 없습니다: {label_encoder_path}")
with open(label_encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

num_labels = len(label_encoder.classes_)

# (6-2) 토크나이저 / 모델 로드
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

bert_tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-multilingual-cased"
)
bert_model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-multilingual-cased",
    num_labels=num_labels
)
# 학습된 가중치 로드
bert_model.load_state_dict(
    torch.load("bert_model.pt", map_location=DEVICE),
    strict=True
)
bert_model.to(DEVICE).eval()


def predict_ost_from_text(scene_text: str) -> str:
    """
    주어진 scene 설명 텍스트(scene_text)를 BERT 모델에 전달해 ost 라벨(“아티스트 - 곡명”) 반환
    """
    bert_model.eval()
    inputs = bert_tokenizer(
        scene_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    ).to(DEVICE)
    with torch.no_grad():
        logits = bert_model(**inputs).logits
        pred_idx = logits.argmax(dim=1).item()
    return label_encoder.inverse_transform([pred_idx])[0]

# (7) CNN 이미지 기반 OST 예측 모델 로드

# (7-1) 이미지 전처리
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# (7-2) CNNClassifier 정의 및 로드
class CNNClassifier(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_labels)

    def forward(self, x):
        return self.backbone(x)

cnn_model = CNNClassifier(num_labels=num_labels)
cnn_model.load_state_dict(
    torch.load("cnn_model.pt", map_location=DEVICE)
)
cnn_model.to(DEVICE).eval()


def predict_ost_from_image(image_path: str) -> str:
    """
    주어진 scene 이미지 경로(image_path)를 CNN 모델에 전달해 ost 라벨(“아티스트 - 곡명”) 반환
    """
    img = Image.open(image_path).convert("RGB")
    tensor = image_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = cnn_model(tensor)
        pred_idx = logits.argmax(dim=1).item()
    return label_encoder.inverse_transform([pred_idx])[0]


# (8) 메인 (CLI)에서 기능 선택 후 실행

if __name__ == "__main__":
    print("==============================================")
    print("   DeepTune 음원 인식 및 드라마 OST 예측 도구")
    print("==============================================")
    print("1) 마이크로 녹음 → Shazam(AudD) 기반 음악 인식")
    print("2) 텍스트(scene 설명) → BERT 기반 드라마 OST 예측")
    print("3) 이미지(scene jpg) → CNN 기반 드라마 OST 예측")
    print("q) 종료")
    print("----------------------------------------------")

    while True:
        choice = input("원하는 기능 번호를 입력하세요 (1/2/3/q): ").strip().lower()
        if choice == "q":
            print("프로그램을 종료합니다.")
            sys.exit(0)

        # ─────────────────────────────────────────────────────────────────
        if choice == "1":
            # Shazam 기능 (AudD + Spotify)
            mp3_file = record_audio_mp3(duration=8)
            audd_res = recognize_with_audd(mp3_file)
            if audd_res is not None:
                try:
                    title  = audd_res['result']['title']
                    artist = audd_res['result']['artist']
                    print(f"[🎵] 인식된 곡: {artist} - {title}")

                    spotify_info = search_spotify(f"{title} {artist}")
                    if spotify_info:
                        print("\n=== Spotify 정보 ===")
                        print(f"제목      : {spotify_info['title']}")
                        print(f"아티스트  : {spotify_info['artist']}")
                        print(f"앨범      : {spotify_info['album']}")
                        print(f"미리듣기  : {spotify_info['preview']}")
                        print(f"앨범커버  : {spotify_info['cover']}")
                        print("======================\n")
                    else:
                        print("[❌] Spotify에서 정보를 찾을 수 없습니다.\n")
                except Exception as e:
                    print(f"[❌] 음악 인식 실패: {e}\n")
            else:
                print("[❌] AudD 인식 실패\n")

        # ─────────────────────────────────────────────────────────────────
        elif choice == "2":
            # BERT 기반 텍스트(scene) → OST 예측
            txt = input("▶ 장면 설명을 입력하세요: ").strip()
            if not txt:
                print("[⚠️] 빈 문자열이 입력됐습니다.\n")
                continue
            predicted = predict_ost_from_text(txt)
            print(f"[🎬] 예측된 OST: {predicted}")

            # Spotify 정보까지 가져오기
            sp_info = search_spotify(predicted)
            if sp_info:
                print("\n=== Spotify 정보 ===")
                print(f"제목      : {sp_info['title']}")
                print(f"아티스트  : {sp_info['artist']}")
                print(f"앨범      : {sp_info['album']}")
                print(f"미리듣기  : {sp_info['preview']}")
                print(f"앨범커버  : {sp_info['cover']}")
                print("======================\n")
            else:
                print("[❌] Spotify에서 정보를 찾을 수 없습니다.\n")

        elif choice == "3":
            # CNN 기반 이미지(scene jpg) → OST 예측
            img_path = input("▶ 예측할 이미지 파일 경로를 입력하세요: ").strip()
            if not os.path.exists(img_path):
                print(f"[❌] 이미지 파일이 없습니다: {img_path}\n")
                continue

            predicted = predict_ost_from_image(img_path)
            print(f"[🖼️] 예측된 OST: {predicted}")

            # Spotify 정보까지 가져오기
            sp_info = search_spotify(predicted)
            if sp_info:
                print("\n=== Spotify 정보 ===")
                print(f"제목      : {sp_info['title']}")
                print(f"아티스트  : {sp_info['artist']}")
                print(f"앨범      : {sp_info['album']}")
                print(f"미리듣기  : {sp_info['preview']}")
                print(f"앨범커버  : {sp_info['cover']}")
                print("======================\n")
            else:
                print("[❌] Spotify에서 정보를 찾을 수 없습니다.\n")

        else:
            print("[⚠️] 잘못된 입력입니다. 다시 시도해주세요.\n")
