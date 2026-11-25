"""화면에서 쓰는 계산 로직 (날씨 수집 → 캠퍼스 체감 → 옷차림 분류 → 결과 묶기)."""
from __future__ import annotations

import math
import os
import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests

import config
import model
import outfit
from feels_like import campus_batch, campus_feels_like, deployed_model_status, model_error
from weather import fetch_campus_now, get_forecast, nowcast_ref

WEEKDAYS = "월화수목금토일"


def r0(x: float) -> int:
    """사사오입 반올림 (화면 JS의 Math.round 와 같게 — 파이썬 round는 8.5→8)."""
    return math.floor(x + 0.5)
_cache = {"fc": None, "at": 0.0, "now": None, "now_at": 0.0, "now_err": None, "nowcast": None}
_model = {"clf": None, "loaded": False}


def forecast():
    if _cache["fc"] is None or time.time() - _cache["at"] > 600:
        _cache["fc"] = get_forecast()
        _cache["at"] = time.time()
    return _cache["fc"]


def campus_now():
    """안양대 500m 격자 최신 실측 (10분 캐시). 키가 없거나 실패하면 None."""
    if time.time() - _cache["now_at"] > 600:
        _cache["now_at"] = time.time()
        try:
            _cache["now"], _cache["now_err"] = fetch_campus_now(), None
        except Exception as e:
            _cache["now"], _cache["now_err"] = None, str(e)
        try:   # 오늘 실측 반영 기준 (정시 실측 + 그 시각 예보)
            _cache["nowcast"] = nowcast_ref(_cache["now"]["hour_obs"]) if _cache["now"] else None
        except Exception:
            _cache["nowcast"] = None
    return _cache["now"]


def clf():
    if not _model["loaded"]:
        _model["clf"] = model.load_model()
        _model["loaded"] = True
    return _model["clf"]


def korean_datetime(dt: datetime) -> str:
    ampm = "오전" if dt.hour < 12 else "오후"
    h12 = dt.hour % 12 or 12
    return f"{dt.month}월 {dt.day}일 ({WEEKDAYS[dt.weekday()]}) {ampm} {h12}:{dt.minute:02d}"


# ---------------- 학생 체감 제보 ----------------

def _feedback_today() -> pd.DataFrame:
    if not config.FEEDBACK_PATH.exists():
        return pd.DataFrame()
    fb = pd.read_csv(config.FEEDBACK_PATH, encoding="utf-8-sig")
    fb["reported_at"] = pd.to_datetime(fb["reported_at"])
    return fb[fb["reported_at"].dt.date == date.today()]


def feedback_stats() -> dict:
    fb = _feedback_today()
    total = len(fb)
    counts = {c: int((fb["feedback"] == c).sum()) if total else 0 for c in outfit.FEEDBACK_CHOICES}
    recent = []
    rows = fb.sort_values("reported_at", ascending=False).head(3).itertuples() if total else []
    for r in rows:
        mins = int((datetime.now() - r.reported_at.to_pydatetime()).total_seconds() // 60)
        ago = "방금" if mins < 1 else (f"{mins}분 전" if mins < 60 else f"{mins // 60}시간 전")
        worn = outfit.get(int(r.worn_level))
        recent.append({"text": f"{worn.items[0]} 입었는데 {r.feedback}", "ago": ago})
    return {
        "total": total,
        "counts": counts,
        "pct": {c: round(100 * n / total) if total else 0 for c, n in counts.items()},
        "recent": recent,
    }


def save_feedback(worn_level: int, feedback: str, sensitivity: str):
    fc = forecast()
    h = fc.hourly
    idx = (h["time"] - pd.Timestamp(datetime.now())).abs().idxmin()
    r = h.loc[idx]
    cf = campus_feels_like(r.temp, r.wind, r.humidity, r.time, sensitivity)
    model.save_feedback(model.make_row(cf, r["pop"], r.time, sensitivity), worn_level, feedback)


# ---------------- 홈 ----------------

def home() -> dict:
    fc = forecast()
    now = datetime.now()
    h = fc.hourly
    if fc.now:
        app_now, label = fc.now["temp"], "기상청 초단기실황"
    else:
        app_now = float(h.loc[(h["time"] - pd.Timestamp(now)).abs().idxmin(), "temp"])
        label = fc.source

    errors = list(fc.errors)
    if model_error():
        errors.append(model_error())
    obs = campus_now()
    if obs is None and os.environ.get("KMA_APIHUB_KEY") and _cache["now_err"]:
        errors.append(f"캠퍼스 실측값을 받지 못했어요: {_cache['now_err']}")
    if obs is not None:   # 안양대 500m 격자 실측 (약 1시간 지연)
        t = pd.Timestamp(obs["time"])
        now_out = {"feels": round(obs["feels"], 1), "temp": round(obs["temp"], 1),
                   "wind": round(obs["wind"], 1), "humidity": round(obs["humidity"]),
                   "time": f"{t:%H:%M}", "kind": "실측"}
    else:                 # 실측이 없으면 지금 시각 예보 → 캠퍼스 AI 예측
        h = fc.hourly
        r = h.loc[(h["time"] - pd.Timestamp(now)).abs().idxmin()]
        campus_predictions = campus_batch(h, _cache["nowcast"])
        predicted = campus_predictions.loc[r.name].to_dict()
        cf = campus_feels_like(r.temp, r.wind, r.humidity, r.time, predicted=predicted)
        now_out = {"feels": round(cf.feels, 1), "temp": round(cf.campus_temp, 1),
                   "wind": round(cf.campus_wind, 1), "humidity": round(cf.campus_humidity),
                   "time": f"{pd.Timestamp(r.time):%H:%M}", "kind": "AI 예측"}
    deployed = deployed_model_status()
    return {
        "campus": config.CAMPUS_NAME,
        "now_label": korean_datetime(now),
        "app_temp": r0(app_now),
        "app_label": label,
        "source": fc.source,
        "errors": errors,
        "app_ref": config.APP_REF_NAME,
        "campus_now": now_out,
        "schedules": list(config.SCHEDULE_PRESETS),
        "sensitivities": list(config.SENSITIVITY_OFFSET),
        "levels": [{"level": l.level, "name": l.name, "items": list(l.items)} for l in outfit.LEVELS],
        "stats": feedback_stats(),
        "data": {
            "campus_model": (
                f"{deployed.get('selected_model_name')} · "
                f"{deployed.get('validation_year')}년 검증 선택"
                if deployed.get("deployed")
                else "랜덤포레스트 · 기존 앱 모델"
            ),
            "campus_obs": "안양대 500m 격자 실측 (기상청, 약 1시간 지연)",
        },
    }


# ---------------- AI 분석 ----------------

def _ai_comment(app_t, feels_t, worst_hour, cats, hourly):
    diff = r0(app_t - feels_t)
    if diff >= 1:
        head = f"날씨앱만 보고 얇게 입으면 후회해요! 캠퍼스는 날씨앱보다 {diff}° 더 춥게 느껴져요. "
    elif diff <= -1:
        head = f"오늘 캠퍼스는 날씨앱보다 {-diff}° 덜 춥게 느껴져요. "
    else:
        head = "오늘 캠퍼스 체감은 날씨앱과 비슷해요. "
    fallback = (
        f"{head}{worst_hour}시가 체감 {feels_t:.0f}°로 가장 추우니 "
        f"{cats['상의'][0]}에 {cats['의류 잡화'][0]}까지 챙기세요."
    )
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return fallback
    try:
        prompt = (
            "너는 안양대 학생에게 옷차림을 알려주는 친근한 도우미야. 존댓말로 2문장 이내.\n"
            f"날씨앱 체감 {app_t:.1f}°C, 안양대 캠퍼스 체감 {feels_t:.1f}°C ({worst_hour}시 가장 추움), "
            f"시간대별 체감: {[(h['hour'], h['feels']) for h in hourly]}, 추천 옷: {cats}"
        )
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}]},
            timeout=15,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return fallback


def analyze(day: str, schedule: str, sensitivity: str) -> dict:
    fc = forecast()
    target = date.today() + timedelta(days=0 if day == "오늘" else 1)
    start_h, end_h = config.SCHEDULE_PRESETS.get(schedule, (9, 18))

    h = fc.hourly
    h = h.sort_values("time").reset_index(drop=True)
    campus_all = campus_batch(h, _cache["nowcast"])
    sel = h[(h["time"].dt.date == target) & (h["time"].dt.hour >= start_h) & (h["time"].dt.hour <= end_h)]
    if day == "오늘":
        sel = sel[sel["time"] >= pd.Timestamp(datetime.now()).floor("h")]
    if sel.empty:
        return {"error": "선택한 시간대의 예보가 없어요. 오늘 일정이 이미 끝났다면 '내일'을 선택하세요."}

    m = clf()
    campus = campus_all.loc[sel.index].reset_index(drop=True)
    sel = sel.reset_index(drop=True)
    obs = campus_now()                       # 지난 시각: 500m 격자 실측
    obs_hour = pd.Timestamp(obs["time"]).floor("h") if obs is not None else None
    hours = []
    for i, r in enumerate(sel.itertuples(index=False)):
        observed = obs if obs_hour is not None and r.time == obs_hour else None
        cf = campus_feels_like(r.temp, r.wind, r.humidity, r.time, sensitivity,
                               observed=observed, predicted=campus.iloc[i])
        row = model.make_row(cf, r.pop, r.time, sensitivity)
        level, _ = model.predict_level(m, row)
        hours.append({"hour": r.time.hour, "app": r.temp, "wind": r.wind, "pop": r.pop,
                      "feels": cf.personal, "level": level, "cf": cf})

    # 하루 종일 버틸 수 있게: 가장 두껍게 입어야 하는 시간대 기준
    worst = max(hours, key=lambda x: (x["level"], -x["feels"]))
    cf = worst["cf"]
    lv = outfit.get(worst["level"])
    levels = [x["level"] for x in hours]
    layering = max(levels) - min(levels) >= 2
    rain_max = max(x["pop"] for x in hours)
    cats = outfit.recommend(lv.level, rain=rain_max >= 60)

    if rain_max >= 60:
        tip = f"강수확률이 최대 {rain_max:.0f}%예요. 우산을 꼭 챙기세요."
    elif layering:
        tip = "일교차가 커요. 벗기 쉬운 겉옷으로 겹쳐 입고 낮에는 한 겹 벗으세요."
    elif cf.wind_delta <= -1:
        tip = "건물 사이 바람이 강해요. 목을 감싸는 옷이 체감온도를 크게 올려줘요."
    else:
        tip = "실내 강의실은 따뜻하니 안에 너무 두껍게 입기보다 겉옷으로 조절하세요."

    # 차트: 범위가 길면 3시간 간격
    chart = hours if len(hours) <= 6 else [x for i, x in enumerate(hours) if i % 3 == 0 or i == len(hours) - 1]
    warning = None
    if len(hours) >= 2 and hours[-1]["feels"] <= min(x["feels"] for x in hours) and hours[0]["feels"] - hours[-1]["feels"] >= 3:
        warning = f"{hours[-1]['hour']}시 이후 기온 급하강 주의"

    stats = feedback_stats()
    first = hours[0]
    errors = list(fc.errors) + ([model_error()] if model_error() else [])
    low, high = cf.feels_low + cf.sensitivity_delta, cf.feels_high + cf.sensitivity_delta
    hourly_out = [{"hour": x["hour"], "app": round(x["app"], 1), "feels": round(x["feels"], 1)} for x in chart]
    return {
        "source": fc.source,
        "errors": errors,
        "date_label": f"{target.month}월 {target.day}일 ({WEEKDAYS[target.weekday()]})",
        "schedule": schedule,
        "end_hour": end_h,
        "steps": {
            "weather": f"{first['app']:.0f}° · {first['wind']:.1f}m/s",
            "campus": f"{cf.climate_delta:+.1f}°",
            "wind": f"{cf.wind_delta:+.1f}°",
            "nowcast": (f"{pd.Timestamp(_cache['nowcast']['time']):%H}시 실측 {cf.nowcast_delta:+.1f}°"
                        if _cache["nowcast"] else "실측 없음"),
            "feedback": f"{stats['total']}건 반영",
            "sensitivity": f"{cf.sensitivity_delta:+.1f}°",
        },
        "result": {
            "hour": worst["hour"],
            "feels": round(cf.personal, 1),  # 소수 첫째 자리까지
            "app": round(cf.app_feels, 1),
            "diff": r0(cf.app_feels - cf.personal),
            "campus": round(cf.climate_delta, 1),
            "wind": round(cf.wind_delta, 1),
            "nowcast": round(cf.nowcast_delta, 1),
            "nowcast_hour": (pd.Timestamp(_cache["nowcast"]["time"]).hour
                             if _cache["nowcast"] and hours[0]["cf"].source.endswith("실측 반영") else None),
            "range": [r0(low), r0(high)] if r0(low) != r0(high) else None,
            "basis": cf.source,
            "sensitivity": round(cf.sensitivity_delta, 1),
            "level": lv.level,
            "level_name": lv.name,
            "categories": [
                {"name": c, "emoji": outfit.CATEGORY_EMOJI[c],
                 "items": [{"name": i, "emoji": outfit.ITEM_EMOJI.get(i, "👕")} for i in items]}
                for c, items in cats.items()
            ],
            "comment": _ai_comment(cf.app_feels, cf.personal, worst["hour"], cats, hourly_out),
            "tip": tip,
            "model": type(m).__name__ if m is not None else "규칙 기반",
            "feedback_total": stats["total"],
        },
        "hourly": hourly_out,
        "warning": warning,
        "stats": stats,
    }
