"""안양대 캠퍼스 날씨 예측 모델 (랜덤포레스트).

만안(안양만안 AWS 377 관측값 또는 만안구 단기예보)의 기온·습도·풍속과 월·시간대를 넣으면
안양대 500m 격자 기준 기온·습도·풍속을 예측하고, 기상청 공식으로 체감온도를 계산한다.

- 모델은 안양대 값을 통째로 맞히지 않고 '안양대 − 만안' 차이(보정량)를 정답마다 따로 학습한다.
- 체감온도는 기상청 공식을 그대로 따른다 (고도 보정 없음).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# 입력 (만안 값 + 시각)
FEATURES = ["aws_ta", "aws_hm", "aws_ws", "hour_sin", "hour_cos", "month_sin", "month_cos"]
# 정답 (안양대 값) 과 각 정답의 기준이 되는 만안 값
TARGETS = ["uni_ta", "uni_hm", "uni_ws"]
BASES = ["aws_ta", "aws_hm", "aws_ws"]

# 나무 100그루, 잎 최소 20개, 깊이 14: 300그루와 정확도는 같고 파일은 약 1/10 (GitHub 100MB 제한 대비)
RF_PARAMS = dict(n_estimators=100, min_samples_leaf=20, max_depth=14, n_jobs=-1, random_state=0)


# ---------------- 기상청 체감온도 공식 ----------------

def kma_winter(ta, ws):
    """겨울철(10~4월): 기온 10°C 이하, 풍속 1.3 m/s 이상일 때만 계산. 아니면 기온 그대로."""
    ta, ws = np.asarray(ta, dtype=float), np.asarray(ws, dtype=float)
    v16 = (ws * 3.6) ** 0.16  # m/s -> km/h
    wc = 13.12 + 0.6215 * ta - 11.37 * v16 + 0.3965 * v16 * ta
    return np.where((ta <= 10) & (ws >= 1.3), wc, ta)


def kma_summer(ta, rh):
    """여름철(5~9월): 기온·습도. 습구온도는 Stull 추정식."""
    ta, rh = np.asarray(ta, dtype=float), np.asarray(rh, dtype=float)
    tw = (ta * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + np.arctan(ta + rh)
          - np.arctan(rh - 1.67633) + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) - 4.686035)
    return -0.2442 + 0.55399 * tw + 0.45535 * ta - 0.0022 * tw ** 2 + 0.00278 * tw * ta + 3.0


def kma_feels(month, ta, rh, ws):
    """관측 월에 따라 여름(5~9월)/겨울(10~4월) 공식을 골라 계산."""
    month = np.asarray(month)
    return np.where((month >= 5) & (month <= 9), kma_summer(ta, rh), kma_winter(ta, ws))


# ---------------- 입력 만들기 ----------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """시간대·월은 순환하므로(23시 다음 0시) sin·cos로 바꾼다."""
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def tidy(pred: np.ndarray) -> np.ndarray:
    """예측값을 말이 되는 범위로: 습도 0~100%, 풍속 0 이상."""
    pred = np.array(pred, dtype=float)
    pred[:, 1] = pred[:, 1].clip(0, 100)
    pred[:, 2] = pred[:, 2].clip(0, None)
    return pred


# ---------------- 모델 ----------------

class CampusModel:
    """정답(기온·습도·풍속)마다 랜덤포레스트 하나: (안양대 − 만안) 보정량을 학습."""

    def __init__(self, **rf_params):
        self.rf_params = {**RF_PARAMS, **rf_params}
        self.models: dict[str, RandomForestRegressor] = {}

    def fit(self, df: pd.DataFrame) -> "CampusModel":
        df = add_features(df)
        for target, base in zip(TARGETS, BASES):
            rf = RandomForestRegressor(**self.rf_params)
            self.models[target] = rf.fit(df[FEATURES], df[target] - df[base])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """df: aws_ta, aws_hm, aws_ws, month, hour → [[안양대 기온, 습도, 풍속], ...]"""
        df = add_features(df)
        delta = np.column_stack([self.models[t].predict(df[FEATURES]) for t in TARGETS])
        return tidy(df[BASES].to_numpy(dtype=float) + delta)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: str | Path) -> "CampusModel":
        return joblib.load(path)


@dataclass
class CampusForecast:
    app_feels: float     # 만안(날씨앱) 값 그대로 계산한 체감온도
    temp: float          # 안양대 예측 기온
    humidity: float      # 안양대 예측 습도
    wind: float          # 안양대 예측 풍속
    feels: float         # 안양대 예측 체감온도
    feels_low: float     # 풍속이 1.3 m/s 기준선 근처일 때의 범위 (아니면 feels와 같음)
    feels_high: float


def forecast_campus(model: CampusModel, when, temp: float, humidity: float, wind: float,
                    band: float = 0.3) -> CampusForecast:
    """만안 예보값 하나 → 안양대 예측.

    겨울에 예측 풍속이 1.3 m/s ± band 안이면, 기준선 양쪽으로 계산한 체감온도를 범위로 돌려준다.
    """
    when = pd.Timestamp(when)
    x = pd.DataFrame({"aws_ta": [temp], "aws_hm": [humidity], "aws_ws": [wind],
                      "month": [when.month], "hour": [when.hour]})
    ta, hm, ws = model.predict(x)[0]
    feels = float(kma_feels(when.month, ta, hm, ws))
    low = high = feels
    if not 5 <= when.month <= 9 and ta <= 10 and abs(ws - 1.3) <= band:
        a = float(kma_feels(when.month, ta, hm, 1.3 + band))   # 바람이 조금 더 셀 때
        b = float(kma_feels(when.month, ta, hm, 1.3 - band))   # 바람이 조금 더 약할 때 (= 기온)
        low, high = min(a, b, feels), max(a, b, feels)
    return CampusForecast(
        app_feels=float(kma_feels(when.month, temp, humidity, wind)),
        temp=float(ta), humidity=float(hm), wind=float(ws),
        feels=feels, feels_low=low, feels_high=high,
    )
