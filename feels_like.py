"""체감온도 (기상청 공식) + 안양대 캠퍼스 보정 (AI 모델).

- 체감온도 공식은 기상청 기준 그대로 (campus_model.kma_feels): 10~4월 바람, 5~9월 습도.
- 캠퍼스 보정은 고도·바람 배율 같은 추정 계수 없이, 학습한 랜덤포레스트가
  만안 날씨(예보)로 안양대 기온·습도·풍속을 예측한다.
- 이미 지난 시각은 안양대 500m 격자 실측값(observed)을 그대로 쓴다.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import config
from campus_model import CampusModel, kma_feels

WIND_LINE = 1.3   # 기상청 겨울 체감온도 계산 기준 풍속 (m/s)
WIND_BAND = 0.3   # 예측 풍속이 기준선 ± 이 범위면 체감온도를 범위로 보여줌

_model = {"m": None, "loaded": False, "error": None}


def campus_model() -> CampusModel | None:
    """data/campus_model.joblib 불러오기. 없으면 data/train_data.csv로 바로 학습 (수 초)."""
    if _model["loaded"]:
        return _model["m"]
    _model["loaded"] = True
    try:
        if not config.CAMPUS_MODEL_PATH.exists():
            if not config.CAMPUS_TRAIN_PATH.exists():
                raise FileNotFoundError("data/train_data.csv 가 없어 캠퍼스 모델을 만들 수 없어요")
            from train_campus import load_data
            print("캠퍼스 모델이 없어 학습합니다 (data/train_data.csv, 수 초)…")
            CampusModel().fit(load_data(config.CAMPUS_TRAIN_PATH)).save(config.CAMPUS_MODEL_PATH)
        m = CampusModel.load(config.CAMPUS_MODEL_PATH)
        for rf in m.models.values():
            rf.n_jobs = 1   # 한두 줄 예측은 단일 스레드가 더 빠름
        _model["m"] = m
    except Exception as e:  # 모델이 없어도 사이트는 동작 (보정 없이)
        _model["error"] = f"캠퍼스 AI 모델을 불러오지 못해 보정 없이 계산해요: {e}"
    return _model["m"]


def model_error() -> str | None:
    campus_model()
    return _model["error"]


def feels_like(temp_c, wind_ms, rh, month):
    """기상청 체감온도 (월에 따라 여름/겨울 공식)."""
    return float(kma_feels(month, temp_c, rh, wind_ms))


def _wind_range(month, ta, rh, ws):
    """겨울에 풍속이 기준선(1.3 m/s) 근처면, 바람이 조금 세거나 약할 때의 체감온도 범위."""
    f = feels_like(ta, ws, rh, month)
    if 5 <= month <= 9 or ta > 10 or abs(ws - WIND_LINE) > WIND_BAND:
        return f, f
    a = feels_like(ta, WIND_LINE + WIND_BAND, rh, month)
    b = feels_like(ta, WIND_LINE - WIND_BAND, rh, month)
    return min(a, b, f), max(a, b, f)


def campus_batch(df: pd.DataFrame) -> pd.DataFrame:
    """여러 시각 한꺼번에: df(time, temp, wind, humidity) → 안양대 기온·습도·풍속 예측.

    모델이 없으면 만안 값을 그대로 쓴다 (보정 없음).
    """
    m = campus_model()
    x = pd.DataFrame({"aws_ta": df["temp"].to_numpy(float), "aws_hm": df["humidity"].to_numpy(float),
                      "aws_ws": df["wind"].to_numpy(float),
                      "month": df["time"].dt.month.to_numpy(), "hour": df["time"].dt.hour.to_numpy()})
    p = m.predict(x) if m is not None else x[["aws_ta", "aws_hm", "aws_ws"]].to_numpy(float)
    return pd.DataFrame({"campus_temp": p[:, 0], "campus_humidity": p[:, 1], "campus_wind": p[:, 2]},
                        index=df.index)


@dataclass
class CampusFeel:
    app_temp: float           # 날씨앱(만안) 기온
    app_feels: float          # 날씨앱 값으로 계산한 체감온도 (같은 기상청 공식)
    campus_temp: float        # 안양대 기온 (예측 또는 실측)
    campus_humidity: float    # 안양대 습도
    campus_wind: float        # 안양대 풍속
    feels: float              # 안양대 체감온도 (개인 민감도 전)
    feels_low: float          # 풍속이 기준선 근처일 때의 범위 (아니면 feels와 같음)
    feels_high: float
    personal: float           # 개인 민감도까지 반영한 최종 체감온도
    climate_delta: float      # 기온·습도 차이로 바뀐 체감 (°C)
    wind_delta: float         # 바람 차이로 바뀐 체감 (°C)
    sensitivity_delta: float  # 개인 보정 (°C)
    source: str               # "AI 예측" | "실측(500m 격자)" | "보정 없음"


def campus_feels_like(temp_c: float, wind_ms: float, rh: float, when, sensitivity: str = "보통",
                      observed: dict | None = None, predicted: dict | None = None) -> CampusFeel:
    """날씨앱(만안) 값 → 안양대 캠퍼스 체감온도.

    observed : 그 시각 안양대 500m 격자 실측값 {temp, humidity, wind} — 있으면 그대로 사용
    predicted: campus_batch 결과 한 줄 (미리 계산해 둔 예측) — 없으면 여기서 예측
    """
    when = pd.Timestamp(when)
    month = when.month
    if observed is not None:
        ta, hm, ws, source = observed["temp"], observed["humidity"], observed["wind"], "실측(500m 격자)"
    else:
        if predicted is None:
            row = pd.DataFrame({"time": [when], "temp": [temp_c], "wind": [wind_ms], "humidity": [rh]})
            predicted = campus_batch(row).iloc[0].to_dict()
        ta, hm, ws = predicted["campus_temp"], predicted["campus_humidity"], predicted["campus_wind"]
        source = "AI 예측" if campus_model() is not None else "보정 없음"

    app_feels = feels_like(temp_c, wind_ms, rh, month)
    feels = feels_like(ta, ws, hm, month)
    mid = feels_like(ta, wind_ms, hm, month)   # 기온·습도만 캠퍼스 값, 바람은 날씨앱 그대로
    low, high = _wind_range(month, ta, hm, ws) if observed is None else (feels, feels)
    sens = config.SENSITIVITY_OFFSET.get(sensitivity, 0.0)
    return CampusFeel(
        app_temp=float(temp_c), app_feels=app_feels,
        campus_temp=float(ta), campus_humidity=float(hm), campus_wind=float(ws),
        feels=feels, feels_low=low, feels_high=high, personal=feels + sens,
        climate_delta=mid - app_feels, wind_delta=feels - mid, sensitivity_delta=sens,
        source=source,
    )
