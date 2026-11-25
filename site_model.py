"""사이트(브라우저)와 똑같은 캠퍼스 체감 계산을 파이썬으로 (예측 기록용).

docs/model.js의 선택 모델을 읽어 standalone.js와 같은 순서로 계산한다.
그래서 GitHub Actions에 기록되는 예측과 학생이 정적 화면에서 본 값이 일치한다.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd

import config
from campus_model import BASES, FEATURES, TARGETS, add_features, tidy
from compare_models import lag_features
from static_neural import neural_delta

NOWCAST_TAU = 4.0


@lru_cache(maxsize=1)
def _campus_model() -> dict:
    text = (config.BASE_DIR / "docs" / "model.js").read_text(encoding="utf-8")
    js = json.loads(text[text.index("{"): text.rindex("}") + 1])
    model = js["campus_model"]
    model.setdefault("kind", "rf")
    model.setdefault("fallback", model)
    return model


def _rf_delta(trees, x) -> float:
    s = 0.0
    for tr in trees:
        n = 0
        while tr["f"][n] != -1:
            n = tr["l"][n] if x[tr["f"][n]] <= tr["t"][n] else tr["r"][n]
        v = tr["v"]
        s += v[str(n)] if isinstance(v, dict) else v[n]   # 잎 값만 {노드번호: 값}으로 저장됨
    return s / len(trees)


def _lgb_value(node, x) -> float:
    while "v" not in node:
        value = x[node["f"]]
        node = (node["l"] if node.get("d", True) else node["r"]) if not np.isfinite(value) \
            else (node["l"] if value <= node["t"] else node["r"])
    return float(node["v"])


def _delta(model, x, target) -> float:
    trees = model["trees"][target]
    if model.get("kind") == "lightgbm":
        return sum(_lgb_value(tree, x) for tree in trees)
    return _rf_delta(trees, x)


def _current_features(t, w, rh, month, hour):
    return [t, rh, w, math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
            math.sin(2 * math.pi * month / 12), math.cos(2 * math.pi * month / 12)]


def _predict_with(model, t, w, rh, x) -> dict:
    values = [t + _delta(model, x, "uni_ta"), rh + _delta(model, x, "uni_hm"),
              w + _delta(model, x, "uni_ws")]
    return {"temp": values[0], "humidity": min(100.0, max(0.0, values[1])),
            "wind": max(0.0, values[2])}


def campus_predict(t, w, rh, month, hour) -> dict:
    model = _campus_model()
    point_model = model if model["kind"] == "rf" else model["fallback"]
    return _predict_with(point_model, t, w, rh, _current_features(t, w, rh, month, hour))


def campus_predict_series(rows: list[dict]) -> list[dict]:
    """forecast.json 전체 시계열에 선택 모델을 적용하고 앞 24시간은 RF로 보완한다."""
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    frame["tm"] = pd.to_datetime(frame["date"] + " " + frame["hour"].astype(str) + ":00")
    frame["aws_ta"], frame["aws_hm"], frame["aws_ws"] = frame.temp, frame.humidity, frame.wind
    frame["month"], frame["hour"] = frame.tm.dt.month, frame.tm.dt.hour
    frame = add_features(frame)
    model = _campus_model()
    fallback = model if model["kind"] == "rf" else model["fallback"]
    base = frame[BASES].to_numpy(float)
    current = frame[FEATURES].to_numpy(float)
    prediction = np.asarray([
        [base[i, 0] + _delta(fallback, current[i], "uni_ta"),
         base[i, 1] + _delta(fallback, current[i], "uni_hm"),
         base[i, 2] + _delta(fallback, current[i], "uni_ws")]
        for i in range(len(frame))
    ])
    if model["kind"] in {"rf-lag", "lightgbm"}:
        lag = lag_features(frame)
        features = model["features"]
        valid = lag[features].notna().all(axis=1).to_numpy()
        for i in np.where(valid)[0]:
            x = lag.iloc[i][features].to_numpy(float)
            prediction[i] = base[i] + [_delta(model, x, target) for target in TARGETS]
    elif model["kind"] in {"lstm", "tsmixer"}:
        window = int(model["window"])
        raw = frame[model["features"]].to_numpy(float)
        for i in range(window - 1, len(frame)):
            first = i - window + 1
            if frame.tm.iloc[i] - frame.tm.iloc[first] != pd.Timedelta(hours=window - 1):
                continue
            sequence = raw[first:i + 1]
            if np.isfinite(sequence).all():
                prediction[i] = base[i] + neural_delta(model, sequence)
    return [
        {"temp": row[0], "humidity": row[1], "wind": row[2]}
        for row in tidy(prediction)
    ]


def feels(month, ta, rh, ws) -> float:
    """기상청 체감온도 (campus_model.kma_feels 와 같은 식, GitHub Actions에서 numpy·sklearn 없이 쓰려고 따로 둠)."""
    if 5 <= month <= 9:
        tw = (ta * math.atan(0.151977 * math.sqrt(rh + 8.313659)) + math.atan(ta + rh) - math.atan(rh - 1.67633)
              + 0.00391838 * rh ** 1.5 * math.atan(0.023101 * rh) - 4.686035)
        return -0.2442 + 0.55399 * tw + 0.45535 * ta - 0.0022 * tw * tw + 0.00278 * tw * ta + 3.0
    if ta > 10 or ws < 1.3:
        return float(ta)
    v16 = (ws * 3.6) ** 0.16
    return 13.12 + 0.6215 * ta - 11.37 * v16 + 0.3965 * v16 * ta


def nowcast_error(nc: dict | None) -> dict | None:
    if not nc or not nc.get("obs") or not nc.get("fcst"):
        return None
    t = datetime.strptime(nc["time"], "%Y-%m-%d %H:%M")
    f, o = nc["fcst"], nc["obs"]
    p = campus_predict(f["temp"], f["wind"], f["humidity"], t.month, t.hour)
    return {"time": t, **{k: o[k] - p[k] for k in ("temp", "humidity", "wind")}}


def campus_feels(t, w, rh, when: datetime, err: dict | None = None,
                 predicted: dict | None = None) -> dict:
    """민감도 '보통' 기준 캠퍼스 체감 (오늘 실측 반영 포함) — standalone.js campusFeel 과 같음."""
    c = dict(predicted or campus_predict(t, w, rh, when.month, when.hour))
    if err:
        lead = max(0.0, (when - err["time"]).total_seconds() / 3600)
        k = math.exp(-lead / NOWCAST_TAU)
        c = {"temp": c["temp"] + k * err["temp"],
             "humidity": min(100.0, max(0.0, c["humidity"] + k * err["humidity"])),
             "wind": max(0.0, c["wind"] + k * err["wind"])}
    return {**c, "feels": feels(when.month, c["temp"], c["humidity"], c["wind"])}
