"""사이트(브라우저)와 똑같은 캠퍼스 체감 계산을 파이썬으로 (예측 기록용).

docs/model.js 의 축소판 랜덤포레스트를 그대로 읽어 docs/standalone.js 의 campusPredict ·
nowcastError · campusFeel 과 같은 순서로 계산한다. 그래서 기록되는 예측 = 학생이 화면에서 본 값.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from functools import lru_cache

import config
from campus_model import kma_feels

NOWCAST_TAU = 4.0


@lru_cache(maxsize=1)
def _trees() -> dict:
    text = (config.BASE_DIR / "docs" / "model.js").read_text(encoding="utf-8")
    js = json.loads(text[text.index("{"): text.rindex("}") + 1])
    return js["campus_model"]["trees"]


def _delta(trees, x) -> float:
    s = 0.0
    for tr in trees:
        n = 0
        while tr["f"][n] != -1:
            n = tr["l"][n] if x[tr["f"][n]] <= tr["t"][n] else tr["r"][n]
        v = tr["v"]
        s += v[str(n)] if isinstance(v, dict) else v[n]   # 잎 값만 {노드번호: 값}으로 저장됨
    return s / len(trees)


def campus_predict(t, w, rh, month, hour) -> dict:
    x = [t, rh, w, math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
         math.sin(2 * math.pi * month / 12), math.cos(2 * math.pi * month / 12)]
    tr = _trees()
    return {"temp": t + _delta(tr["uni_ta"], x),
            "humidity": min(100.0, max(0.0, rh + _delta(tr["uni_hm"], x))),
            "wind": max(0.0, w + _delta(tr["uni_ws"], x))}


def feels(month, ta, rh, ws) -> float:
    return float(kma_feels(month, ta, rh, ws))


def nowcast_error(nc: dict | None) -> dict | None:
    if not nc or not nc.get("obs") or not nc.get("fcst"):
        return None
    t = datetime.strptime(nc["time"], "%Y-%m-%d %H:%M")
    f, o = nc["fcst"], nc["obs"]
    p = campus_predict(f["temp"], f["wind"], f["humidity"], t.month, t.hour)
    return {"time": t, **{k: o[k] - p[k] for k in ("temp", "humidity", "wind")}}


def campus_feels(t, w, rh, when: datetime, err: dict | None = None) -> dict:
    """민감도 '보통' 기준 캠퍼스 체감 (오늘 실측 반영 포함) — standalone.js campusFeel 과 같음."""
    c = campus_predict(t, w, rh, when.month, when.hour)
    if err:
        lead = max(0.0, (when - err["time"]).total_seconds() / 3600)
        k = math.exp(-lead / NOWCAST_TAU)
        c = {"temp": c["temp"] + k * err["temp"],
             "humidity": min(100.0, max(0.0, c["humidity"] + k * err["humidity"])),
             "wind": max(0.0, c["wind"] + k * err["wind"])}
    return {**c, "feels": feels(when.month, c["temp"], c["humidity"], c["wind"])}
