"""체감온도 (기상청 공식) + 안양대 캠퍼스 보정 (AI 모델).

- 체감온도 공식은 기상청 기준 그대로 (campus_model.kma_feels): 10~4월 바람, 5~9월 습도.
- 캠퍼스 보정은 고도·바람 배율 같은 추정 계수 없이, 학습한 랜덤포레스트가
  만안 날씨(예보)로 안양대 기온·습도·풍속을 예측한다.
- 이미 지난 시각은 안양대 500m 격자 실측값(observed)을 그대로 쓴다.
- 오늘 실측 반영(nowcast): 최근 정시 실측과, 그 시각 예보로 AI가 낸 예측의 차이(오차)를
  앞으로의 예측에 더한다. 예보 자체가 빗나간 것과 AI가 빗나간 것이 함께 잡힌다.
  오차는 몇 시간 동안 같은 방향으로 이어지므로 가중치 exp(−경과시간/4시간)으로 줄여 간다.
  (2025년 검증: ±1°C 정답률 1시간 뒤 72.0 → 78.9%, 3시간 뒤 74.6%, 6시간 뒤 거의 원래대로)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from campus_model import CampusModel, kma_feels

WIND_LINE = 1.3   # 기상청 겨울 체감온도 계산 기준 풍속 (m/s)
WIND_BAND = 0.3   # 예측 풍속이 기준선 ± 이 범위면 체감온도를 범위로 보여줌
NOWCAST_TAU = 4.0  # 오늘 실측 반영 가중치가 줄어드는 속도 (시간)

_model = {"m": None, "loaded": False, "error": None}


def campus_model():
    """배포된 검증 1위 모델을 우선 사용하고, 없으면 기존 RF를 사용한다."""
    if _model["loaded"]:
        return _model["m"]
    _model["loaded"] = True
    try:
        if config.DEPLOYED_CAMPUS_MANIFEST.is_file():
            from deployed_model import DeployedCampusModel

            _model["m"] = DeployedCampusModel(config.DEPLOYED_CAMPUS_MANIFEST)
            return _model["m"]
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


def deployed_model_status() -> dict:
    from deployed_model import deployment_status

    campus_model()
    status = deployment_status()
    if _model["error"]:
        status["load_error"] = _model["error"]
    return status


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


def _predict(m, temp, hum, wind, times) -> np.ndarray:
    times = pd.to_datetime(pd.Series(times))
    x = pd.DataFrame({"aws_ta": np.asarray(temp, float), "aws_hm": np.asarray(hum, float),
                      "aws_ws": np.asarray(wind, float),
                      "month": times.dt.month.to_numpy(), "hour": times.dt.hour.to_numpy(),
                      "tm": times.to_numpy()})
    return m.predict(x) if m is not None else x[["aws_ta", "aws_hm", "aws_ws"]].to_numpy(float)


def nowcast_error(nowcast: dict | None) -> dict | None:
    """정시 실측 − (그 시각 예보 → AI 예측). nowcast = {time, obs{temp,humidity,wind}, fcst{...}}."""
    if not nowcast:
        return None
    f, o = nowcast["fcst"], nowcast["obs"]
    p = _predict(campus_model(), [f["temp"]], [f["humidity"]], [f["wind"]], [nowcast["time"]])[0]
    return {"time": pd.Timestamp(nowcast["time"]), "temp": o["temp"] - p[0],
            "humidity": o["humidity"] - p[1], "wind": o["wind"] - p[2]}


def nowcast_weight(times, ref_time) -> np.ndarray:
    lead = (pd.to_datetime(pd.Series(times)) - pd.Timestamp(ref_time)).dt.total_seconds() / 3600
    return np.exp(-np.clip(lead.to_numpy(float), 0, None) / NOWCAST_TAU)


def campus_batch(df: pd.DataFrame, nowcast: dict | None = None) -> pd.DataFrame:
    """여러 시각 한꺼번에: df(time, temp, wind, humidity) → 안양대 기온·습도·풍속 예측.

    nowcast(weather.nowcast_ref 결과)가 있으면 오늘 실측 오차를 반영한다.
    base_* 는 반영 전 AI 예측, nowcast_w 는 반영 가중치 (0~1).
    모델이 없으면 만안 값을 그대로 쓴다 (보정 없음).
    """
    p = _predict(campus_model(), df["temp"], df["humidity"], df["wind"], df["time"])
    out = pd.DataFrame({"base_temp": p[:, 0], "base_humidity": p[:, 1], "base_wind": p[:, 2]},
                       index=df.index)
    err = nowcast_error(nowcast)
    w = nowcast_weight(df["time"], err["time"]) if err else np.zeros(len(df))
    e = err or {"temp": 0.0, "humidity": 0.0, "wind": 0.0}
    out["campus_temp"] = out["base_temp"] + w * e["temp"]
    out["campus_humidity"] = np.clip(out["base_humidity"] + w * e["humidity"], 0, 100)
    out["campus_wind"] = np.clip(out["base_wind"] + w * e["wind"], 0, None)
    out["nowcast_w"] = w
    return out


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
    source: str               # "AI 예측" | "AI 예측 + 오늘 실측 반영" | "실측(500m 격자)" | "보정 없음"
    nowcast_delta: float = 0.0  # 오늘 실측 반영으로 바뀐 체감 (°C)


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
    # 오늘 실측 반영 전 값 (기온·습도/바람 분해는 반영 전 AI 예측으로)
    bta = predicted.get("base_temp", ta) if observed is None else ta
    bhm = predicted.get("base_humidity", hm) if observed is None else hm
    bws = predicted.get("base_wind", ws) if observed is None else ws

    app_feels = feels_like(temp_c, wind_ms, rh, month)
    feels = feels_like(ta, ws, hm, month)
    base = feels_like(bta, bws, bhm, month)
    nowcast = feels - base
    if observed is None and predicted.get("nowcast_w", 0) >= 0.05:
        source = "AI 예측 + 오늘 실측 반영"
    mid = feels_like(bta, wind_ms, bhm, month)   # 기온·습도만 캠퍼스 값, 바람은 날씨앱 그대로
    low, high = _wind_range(month, ta, hm, ws) if observed is None else (feels, feels)
    sens = config.SENSITIVITY_OFFSET.get(sensitivity, 0.0)
    return CampusFeel(
        app_temp=float(temp_c), app_feels=app_feels,
        campus_temp=float(ta), campus_humidity=float(hm), campus_wind=float(ws),
        feels=feels, feels_low=low, feels_high=high, personal=feels + sens,
        climate_delta=mid - app_feels, wind_delta=base - mid, sensitivity_delta=sens,
        source=source, nowcast_delta=nowcast,
    )
