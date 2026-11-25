"""옷차림 분류 모델 (Random Forest / XGBoost).

학습 데이터 = 날씨 데이터(기상청 AWS CSV 또는 합성 데이터)에 캠퍼스 보정(AI 모델)을 적용하고
'기온별 옷차림 표' 규칙으로 붙인 초기 라벨 + 학생 체감 제보(feedback.csv)로 고친 라벨.
제보가 쌓일수록 규칙이 아니라 실제 안양대 학생 체감에 맞춰집니다.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import config
import outfit
from feels_like import CampusFeel, campus_batch, campus_feels_like

SENS_CODE = {name: i for i, name in enumerate(config.SENSITIVITY_OFFSET)}  # 많이 타요=0, 보통=1, 안 타요=2
FEATURES = ["campus_temp", "campus_wind", "humidity", "feels", "hour", "month", "sensitivity", "pop"]
FEEDBACK_WEIGHT = 5.0  # 실제 학생 제보는 합성 라벨보다 5배 중요하게


def make_row(cf: CampusFeel, pop, when, sensitivity):
    """캠퍼스 체감 결과 한 시각 → 옷차림 모델 입력."""
    return {
        "campus_temp": cf.campus_temp,
        "campus_wind": cf.campus_wind,
        "humidity": cf.campus_humidity,
        "feels": cf.personal,
        "hour": when.hour,
        "month": when.month,
        "sensitivity": SENS_CODE.get(sensitivity, 1),
        "pop": pop,
    }


# ---------- 학습 데이터 만들기 ----------

def synthetic_weather(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    """안양 기후를 흉내 낸 합성 시간별 날씨 (AWS 자료가 없을 때)."""
    rng = np.random.default_rng(seed)
    monthly_mean = np.array([-2.4, 0.4, 5.7, 12.5, 17.8, 22.2, 25.0, 25.7, 21.2, 14.8, 7.2, 0.4])
    month = rng.integers(1, 13, n)
    hour = rng.integers(7, 23, n)
    temp = monthly_mean[month - 1] + 4.5 * np.sin((hour - 9) / 24 * 2 * np.pi) + rng.normal(0, 3, n)
    return pd.DataFrame({
        "time": pd.to_datetime({"year": 2025, "month": month, "day": 15, "hour": hour}),
        "temp": temp,
        "wind": np.clip(rng.gamma(2.0, 1.2, n), 0, 15),
        "humidity": np.clip(rng.normal(60, 18, n), 15, 100),
        "pop": np.clip(rng.normal(20, 25, n), 0, 100),
    })


def load_aws_csv(path: str | Path) -> pd.DataFrame:
    """기상청 기상자료개방포털 AWS/ASOS 시간자료 CSV 읽기 (열 이름 자동 인식)."""
    df = None
    for enc in ("cp949", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        raise ValueError(f"CSV 인코딩을 읽을 수 없습니다: {path}")

    def find(keyword):
        for c in df.columns:
            if keyword in c:
                return c
        raise KeyError(f"'{keyword}' 열을 찾을 수 없습니다. 열 목록: {list(df.columns)}")

    out = pd.DataFrame({
        "time": pd.to_datetime(df[find("일시")]),
        "temp": df[find("기온")],
        "wind": df[find("풍속")],
        "humidity": df[find("습도")],
    })
    out["pop"] = 0.0
    return out.dropna()


def build_dataset(weather: pd.DataFrame, seed: int = 0):
    """날씨 × 3가지 추위 민감도 → 특성 + 규칙 라벨(사람마다 다른 체감을 노이즈로 반영)."""
    rng = np.random.default_rng(seed)
    weather = weather.reset_index(drop=True)
    campus = campus_batch(weather)   # 캠퍼스 AI 예측은 한꺼번에 (한 줄씩 하면 느림)
    rows, labels = [], []
    for i, r in enumerate(weather.itertuples(index=False)):
        for sens in config.SENSITIVITY_OFFSET:
            cf = campus_feels_like(r.temp, r.wind, r.humidity, r.time, sens, predicted=campus.iloc[i])
            row = make_row(cf, r.pop, r.time, sens)
            rows.append(row)
            labels.append(outfit.rule_level(row["feels"] + rng.normal(0, 1.2)))
    X = pd.DataFrame(rows)[FEATURES]
    y = np.array(labels)
    return X, y, np.ones(len(y))


def load_feedback():
    """학생 체감 제보 → (X, 수정된 라벨, 가중치)."""
    if not config.FEEDBACK_PATH.exists():
        return None
    fb = pd.read_csv(config.FEEDBACK_PATH, encoding="utf-8-sig")
    if fb.empty:
        return None
    y = np.array([outfit.corrected_level(int(l), f) for l, f in zip(fb["worn_level"], fb["feedback"])])
    return fb[FEATURES], y, np.full(len(y), FEEDBACK_WEIGHT)


# ---------- 학습 / 예측 ----------

def make_model(kind: str = "rf"):
    if kind == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, eval_metric="mlogloss")
    return RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=3, random_state=0, n_jobs=-1)


def train(weather: pd.DataFrame, kind: str = "rf"):
    X, y, w = build_dataset(weather)
    fb = load_feedback()
    if fb is not None:
        X = pd.concat([X, fb[0]], ignore_index=True)
        y = np.concatenate([y, fb[1]])
        w = np.concatenate([w, fb[2]])

    X_tr, X_te, y_tr, y_te, w_tr, _ = train_test_split(X, y, w, test_size=0.2, random_state=0, stratify=y)
    model = make_model(kind)
    model.fit(X_tr, y_tr, sample_weight=w_tr)

    pred = model.predict(X_te)
    report = {
        "n_train": len(y_tr),
        "n_feedback": 0 if fb is None else len(fb[1]),
        "accuracy": float((pred == y_te).mean()),
        "within_1": float((np.abs(pred - y_te) <= 1).mean()),
    }
    # 운영용 모델은 전체 데이터로 다시 학습
    model = make_model(kind)
    model.fit(X, y, sample_weight=w)
    config.DATA_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "kind": kind, "features": FEATURES}, config.MODEL_PATH)
    return report


def load_model():
    if not config.MODEL_PATH.exists():
        return None
    return joblib.load(config.MODEL_PATH)["model"]


def predict_level(model, row: dict):
    """(단계, 단계별 확률 dict). 모델이 없으면 규칙 기반."""
    if model is None:
        return outfit.rule_level(row["feels"]), {}
    X = pd.DataFrame([row])[FEATURES]
    proba = model.predict_proba(X)[0]
    classes = [int(c) for c in model.classes_]
    return classes[int(np.argmax(proba))], dict(zip(classes, proba))


def save_feedback(row: dict, worn_level: int, feedback: str):
    config.DATA_DIR.mkdir(exist_ok=True)
    rec = pd.DataFrame([{**row, "worn_level": worn_level, "feedback": feedback,
                         "reported_at": pd.Timestamp.now().isoformat(timespec="seconds")}])
    write_header = not config.FEEDBACK_PATH.exists()
    rec.to_csv(config.FEEDBACK_PATH, mode="a", header=write_header, index=False, encoding="utf-8-sig")
