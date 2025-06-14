📌 DeepTune PyQt5 UI 연동 안내서
-------------------------------

이 파일은 PyQt5로 구현된 DeepTune 데스크탑 UI와 백엔드 Flask 서버 간의 연동을 위한 설명서입니다.

📁 첨부된 파일
-------------
- deeptune_ui.py         → 전체 UI 코드
- assets/                → 배경 및 예시 이미지 (앨범 커버, 버튼 아이콘 등)

🎯 프론트 작동 방식
-------------------
사용자가 다음 중 하나의 입력을 선택:
1. 텍스트 입력 (장면 설명)
2. 이미지 파일 업로드 (스크린샷 등)
3. 마이크 녹음 (5초간 소리 입력)

각 요청은 Flask 백엔드에 전송되며,
응답으로 OST 정보 (제목, 아티스트, 커버 이미지)를 받아 UI에 출력합니다.

📡 백엔드 요청 명세
-------------------

[1] 텍스트 입력
  - URL: /analyze_text
  - Method: POST
  - Body (JSON):
    {
      "text": "장면 설명 텍스트"
    }

[2] 이미지 업로드
  - URL: /analyze_image
  - Method: POST
  - Content-Type: multipart/form-data
  - Field name: "image"

[3] 오디오 업로드 (5초 녹음된 .wav 파일)
  - URL: /analyze_audio
  - Method: POST
  - Content-Type: multipart/form-data
  - Field name: "audio"

📥 프론트가 기대하는 응답 포맷
-------------------------------
모든 요청에 대한 응답은 JSON 형태로 다음 필드를 포함해야 합니다:

{
  "title": "곡 제목",
  "artist": "가수 이름",
  "cover": "이미지 경로 or URL or base64"
}

✔ 코드 실행 전 가상환경 터미널에서 명령어 pip install -r requirements.txt를 입력해 필요한 라이브러리를 다운 받아야 합니다.
✔ 필드 누락 시 UI에서 경고 메시지가 표시됩니다.
✔ cover는 로컬 이미지 경로, 웹 URL, 또는 base64 형식
