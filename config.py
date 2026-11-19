"""오늘춥나?? 설정값.

캠퍼스 보정은 추정 계수(고도·바람 배율)가 아니라 학습한 AI 모델이 합니다.
  - 학습 데이터: data/train_data.csv (안양만안 AWS 377 ↔ 안양대 500m 격자, 2023~2025년 시간별)
  - 모델: python train_campus.py → data/campus_model.joblib
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def load_env(path: Path = BASE_DIR / ".env"):
    """.env 파일의 KEY=VALUE 를 환경변수로 (이미 설정된 값은 유지)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env()

DATA_DIR = BASE_DIR / "data"
MODEL_PATH = DATA_DIR / "outfit_model.joblib"
FEEDBACK_PATH = DATA_DIR / "feedback.csv"

# 안양대학교 안양캠퍼스 (경기도 안양시 만안구 삼덕로37번길 22)
# 500m 격자 학습 데이터를 모은 좌표와 같아야 합니다.
CAMPUS_NAME = "안양대학교 안양캠퍼스"
CAMPUS_LAT = 37.3921
CAMPUS_LON = 126.9206

# '일반 날씨 앱' 기준 지점: 안양만안 AWS(377) — 모델 학습 입력과 같은 만안구 기준
APP_REF_NAME = "안양시 만안구"
APP_REF_LAT = 37.39523
APP_REF_LON = 126.89194

# 기상청 단기예보 격자 좌표 (안양시 만안구)
KMA_NX = 59
KMA_NY = 123

# 캠퍼스 AI 모델 (만안 날씨 → 안양대 기온·습도·풍속)
CAMPUS_TRAIN_PATH = DATA_DIR / "train_data.csv"
CAMPUS_MODEL_PATH = DATA_DIR / "campus_model.joblib"

# 안양대 500m 격자 실시간 값 (기상청 API허브 고해상도 격자 · 특정지점 다중요소, 약 1시간 지연)
CAMPUS_GRID_URL = "https://apihub.kma.go.kr/api/typ01/url/sfc_nc_var.php"

# "나는 추위를..." 개인 민감도 보정 (°C, 체감온도에 더함)
SENSITIVITY_OFFSET = {
    "많이 타요": -2.0,
    "보통": 0.0,
    "안 타요": +2.0,
}

# 캠퍼스에 머무는 시간 프리셋 (시작시, 종료시)
SCHEDULE_PRESETS = {
    "오전만 (09~12시)": (9, 12),
    "09:00 ~ 18:00": (9, 18),
    "야간까지 (09~22시)": (9, 22),
}
