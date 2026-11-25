"""시간대별 날씨 예보 가져오기.

우선순위:
  1) 기상청 — 공공데이터포털(KMA_API_KEY) 또는 기상청 API허브(KMA_APIHUB_KEY)
     · 단기예보(getVilageFcst): 시간대별 기온·풍속·습도·강수확률
     · 초단기실황(getUltraSrtNcst): 지금 이 시각 관측값
  2) Open-Meteo (무료, 키 불필요)
  3) 네트워크가 안 되면 시연용 데모 데이터

캠퍼스 실측: 기상청 API허브 고해상도 격자(500m)로 안양대 위치의 최근 1시간 값 (KMA_APIHUB_KEY 필요)
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from urllib.parse import unquote

import pandas as pd
import requests

import config

COLUMNS = ["time", "temp", "wind", "humidity", "pop"]

DATA_GO_KR = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0"
APIHUB = "https://apihub.kma.go.kr/api/typ02/openApi/VilageFcstInfoService_2.0"


@dataclass
class Forecast:
    hourly: pd.DataFrame
    source: str
    now: Optional[dict] = None              # 초단기실황 {temp, wind, humidity, time}
    errors: list = field(default_factory=list)


# ---------------- 기상청 ----------------

def kst_now() -> datetime:
    """한국 시간 (GitHub Actions 서버처럼 시계가 UTC인 곳에서도 기상청 발표 시각을 맞게 계산)."""
    return datetime.now(timezone(timedelta(hours=9))).replace(tzinfo=None)


def _kma_endpoint():
    """(기본 URL, 인증 파라미터 dict) — 설정된 키 종류에 따라."""
    if os.environ.get("KMA_API_KEY"):
        # 포털의 '인코딩 키'를 넣어도 동작하도록 한 번 디코딩 (requests가 다시 인코딩함)
        return DATA_GO_KR, {"serviceKey": unquote(os.environ["KMA_API_KEY"].strip())}
    if os.environ.get("KMA_APIHUB_KEY"):
        return APIHUB, {"authKey": os.environ["KMA_APIHUB_KEY"].strip()}
    return None, None


def _kma_call(op: str, base_date: str, base_time: str) -> list:
    url, auth = _kma_endpoint()
    resp = requests.get(
        f"{url}/{op}",
        params={**auth, "pageNo": 1, "numOfRows": 1000, "dataType": "JSON",
                "base_date": base_date, "base_time": base_time,
                "nx": config.KMA_NX, "ny": config.KMA_NY},
        timeout=10,
    )
    if resp.status_code != 200:
        # URL에 인증키가 들어 있으므로 상태 코드만 노출
        raise RuntimeError(f"HTTP {resp.status_code} (인증키·활용신청 승인 여부를 확인하세요)")
    try:
        js = resp.json()
    except ValueError:
        # 키 오류 등은 XML로 옴 (예: SERVICE_KEY_IS_NOT_REGISTERED_ERROR)
        raise RuntimeError(f"기상청 응답 오류: {resp.text[:200]}")
    header = js["response"]["header"]
    if header["resultCode"] != "00":
        raise RuntimeError(f"기상청 오류 {header['resultCode']}: {header['resultMsg']}")
    return js["response"]["body"]["items"]["item"]


def _vilage_base(now: datetime):
    """단기예보 발표시각(02,05,...,23시, 발표 10분 후 조회 가능) 중 가장 최근 것."""
    for h in (23, 20, 17, 14, 11, 8, 5, 2):
        if now >= now.replace(hour=h, minute=10, second=0, microsecond=0):
            return now.strftime("%Y%m%d"), f"{h:02d}00"
    return (now - timedelta(days=1)).strftime("%Y%m%d"), "2300"


def _ncst_base(now: datetime):
    """초단기실황: 매시 정각 자료가 약 40분 이후 제공."""
    t = now - timedelta(minutes=40)
    return t.strftime("%Y%m%d"), t.strftime("%H00")


def fetch_kma_forecast() -> pd.DataFrame:
    items = _kma_call("getVilageFcst", *_vilage_base(kst_now()))
    keymap = {"TMP": "temp", "WSD": "wind", "REH": "humidity", "POP": "pop"}
    rows = {}
    for it in items:
        col = keymap.get(it["category"])
        if col:
            t = datetime.strptime(it["fcstDate"] + it["fcstTime"], "%Y%m%d%H%M")
            rows.setdefault(t, {"time": t})[col] = float(it["fcstValue"])
    df = pd.DataFrame(sorted(rows.values(), key=lambda r: r["time"]))
    return df[COLUMNS].dropna()


def _ultra_base(now: datetime):
    """초단기예보: 매시 30분 발표, 약 15분 뒤(45분)부터 조회 가능."""
    t = now - timedelta(minutes=45)
    return t.strftime("%Y%m%d"), t.strftime("%H30")


def fetch_kma_ultra(now: datetime | None = None) -> pd.DataFrame:
    """만안구 초단기예보: 앞으로 6시간, 1시간 간격 (기온·습도·풍속). 매시간 새로 나와 단기예보보다 최신."""
    base_date, base_time = _ultra_base(now or kst_now())
    items = _kma_call("getUltraSrtFcst", base_date, base_time)
    keymap = {"T1H": "temp", "WSD": "wind", "REH": "humidity"}
    rows = {}
    for it in items:
        col = keymap.get(it["category"])
        if col:
            t = datetime.strptime(it["fcstDate"] + it["fcstTime"], "%Y%m%d%H%M")
            rows.setdefault(t, {"time": t})[col] = float(it["fcstValue"])
    df = pd.DataFrame(sorted(rows.values(), key=lambda r: r["time"]))
    df = df[["time", "temp", "wind", "humidity"]].dropna() if not df.empty else df
    df.attrs["base"] = f"{base_date} {base_time}"
    return df


def overlay_ultra(forecast: pd.DataFrame, ultra: pd.DataFrame | None) -> pd.DataFrame:
    """가까운 시간(초단기예보가 있는 최대 6시간)은 기온·습도·풍속을 초단기예보 값으로 바꾼다.

    강수확률은 초단기예보에 없어서 단기예보 값을 그대로 둔다. 먼 시간은 단기예보 그대로.
    """
    if ultra is None or ultra.empty or forecast.empty:
        forecast.attrs["ultra_hours"] = 0
        return forecast
    f = forecast.set_index("time").copy()
    u = ultra.set_index("time")
    common = f.index.intersection(u.index)
    for c in ("temp", "wind", "humidity"):
        f.loc[common, c] = u.loc[common, c]
    out = f.reset_index()
    out.attrs["ultra_hours"] = len(common)
    out.attrs["ultra_base"] = ultra.attrs.get("base", "")
    return out


def fetch_kma_ultra_at(when: datetime) -> dict:
    """그 시각(when, 정시)을 사이트가 직전 실행(약 17분 전)에서 받은 초단기예보로 예측한 값."""
    df = fetch_kma_ultra(when - timedelta(minutes=17))
    row = df[df["time"] == when]
    if row.empty:
        raise RuntimeError(f"{when:%m-%d %H}시가 초단기예보 {df.attrs.get('base')} 발표분에 없음")
    r = row.iloc[0]
    return {"temp": float(r.temp), "wind": float(r.wind), "humidity": float(r.humidity),
            "base": f"{df.attrs.get('base')} 초단기"}


def fetch_kma_at(when: datetime) -> dict:
    """그 시각(when, 정시)에 사이트가 쓰고 있던 만안구 예보값 {temp, wind, humidity, base}.

    when 시점에 받을 수 있었던 가장 최근 단기예보 발표분을 다시 조회한다.
    '오늘 실측 반영'에서 AI 오차(실측 − 예보로 한 예측)를 계산할 때 쓴다.
    """
    base_date, base_time = _vilage_base(when)
    items = _kma_call("getVilageFcst", base_date, base_time)
    want = (when.strftime("%Y%m%d"), when.strftime("%H00"))
    keymap = {"TMP": "temp", "WSD": "wind", "REH": "humidity"}
    v = {keymap[it["category"]]: float(it["fcstValue"]) for it in items
         if it["category"] in keymap and (it["fcstDate"], it["fcstTime"]) == want}
    if len(v) < 3:
        raise RuntimeError(f"{when:%m-%d %H}시 예보가 {base_date} {base_time} 발표분에 없음")
    return {**v, "base": f"{base_date} {base_time}"}


def fetch_kma_now() -> dict:
    base_date, base_time = _ncst_base(kst_now())
    items = _kma_call("getUltraSrtNcst", base_date, base_time)
    v = {it["category"]: float(it["obsrValue"]) for it in items}
    return {
        "temp": v["T1H"], "wind": v.get("WSD", 0.0), "humidity": v.get("REH", 50.0),
        "time": datetime.strptime(base_date + base_time, "%Y%m%d%H%M"),
    }


def fetch_kma_ncst_at(when: datetime) -> dict:
    """만안구 초단기실황: 그 시각(정시)에 실제 관측한 값. 최근 약 24시간까지 조회 가능."""
    items = _kma_call("getUltraSrtNcst", when.strftime("%Y%m%d"), when.strftime("%H00"))
    v = {it["category"]: float(it["obsrValue"]) for it in items}
    if "T1H" not in v:
        raise RuntimeError(f"{when:%m-%d %H}시 초단기실황에 기온 없음")
    return {"temp": v["T1H"], "wind": v.get("WSD", 0.0), "humidity": v.get("REH", 50.0)}


def prepend_observed(forecast: pd.DataFrame, observed: dict, hours: int = 26) -> pd.DataFrame:
    """지연특성용 지난 시간을 붙인다: 만안구 초단기실황 기록(observed) 우선, 빈 시간만 Open-Meteo.

    observed = {"YYYY-MM-DD HH:00": {"temp", "wind", "humidity"}}.
    학습 때 지난 24시간 자리에는 만안 AWS '관측값'이 들어갔으므로, 운영에서도 관측(초단기실황)을 쓴다.
    미래 예보값은 그대로 두고, 결과는 빈 시간 없이 1시간 간격으로 이어지게 만든다(지연특성은 줄 이동으로 계산).
    """
    if forecast.empty:
        return forecast
    start = forecast["time"].min()
    want = pd.date_range(start - timedelta(hours=hours), start - timedelta(hours=1), freq="h")
    rows = {}
    for k, v in (observed or {}).items():
        t = pd.Timestamp(k)
        if t in want:
            rows[t] = {"time": t.to_pydatetime(), "temp": v["temp"], "wind": v["wind"],
                       "humidity": v["humidity"], "pop": 0.0}
    from_obs = set(rows)
    missing = [t for t in want if t not in rows]
    if missing:   # 기록이 빈 시간만 Open-Meteo로
        try:
            om = fetch_open_meteo()
            om = om.set_index(pd.to_datetime(om["time"]))
            for t in missing:
                if t in om.index:
                    r = om.loc[t]
                    rows[t] = {"time": t.to_pydatetime(), "temp": float(r.temp), "wind": float(r.wind),
                               "humidity": float(r.humidity), "pop": 0.0}
        except Exception:
            pass
    past = pd.DataFrame([rows[t] for t in want if t in rows], columns=COLUMNS)
    out = (pd.concat([past, forecast], ignore_index=True)
           .drop_duplicates("time", keep="last").sort_values("time").reset_index(drop=True))
    # 지난 시간 쪽에 빈 시간이 남으면 그 뒤부터만 사용 (줄 이동 지연특성이 어긋나지 않게).
    # 예보 쪽(미래)은 원래대로 둔다 — 먼 미래에 빈 시간이 있어도 지난 기록을 버리지 않도록.
    head = out[out["time"] <= start]
    gaps = head["time"].diff() != pd.Timedelta(hours=1)
    gaps.iloc[0] = False
    if gaps.any():
        out = out.loc[gaps[gaps].index[-1]:].reset_index(drop=True)
    out.attrs["observed_hours"] = int(sum(t in from_obs for t in out["time"]))
    return out


# ---------------- 대체 소스 ----------------

def fetch_open_meteo():
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            # 캠퍼스가 아니라 '날씨 앱 기준 지점'(만안구) 예보를 받아 AI 모델로 캠퍼스 값 예측
            "latitude": config.APP_REF_LAT, "longitude": config.APP_REF_LON,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation_probability",
            "wind_speed_unit": "ms", "timezone": "Asia/Seoul", "past_days": 1, "forecast_days": 3,
        },
        timeout=10,
    )
    resp.raise_for_status()
    js = resp.json()
    h = js["hourly"]
    df = pd.DataFrame({
        "time": pd.to_datetime(h["time"]),
        "temp": h["temperature_2m"],
        "wind": h["wind_speed_10m"],
        "humidity": h["relative_humidity_2m"],
        "pop": h["precipitation_probability"],
    })
    return df.fillna(0)


def prepend_history(forecast: pd.DataFrame) -> pd.DataFrame:
    """지연특성용 지난 24시간을 붙인다. 미래 예보값은 원래 소스를 유지한다."""
    if forecast.empty:
        return forecast
    try:
        context = fetch_open_meteo()
        past = context[context["time"] < forecast["time"].min()]
        return (pd.concat([past, forecast], ignore_index=True)
                .drop_duplicates("time", keep="last").sort_values("time").reset_index(drop=True))
    except Exception:
        return forecast


def demo_forecast(start: date) -> pd.DataFrame:
    """인터넷이 없을 때 시연용 데이터 (월별 평년기온 + 일교차 곡선)."""
    monthly_mean = [-2.4, 0.4, 5.7, 12.5, 17.8, 22.2, 25.0, 25.7, 21.2, 14.8, 7.2, 0.4]
    rows = []
    for d in range(-1, 3):
        day = start + timedelta(days=d)
        mean = monthly_mean[day.month - 1]
        for hour in range(24):
            rows.append({
                "time": datetime(day.year, day.month, day.day, hour),
                "temp": round(mean + 4.5 * math.sin((hour - 9) / 24 * 2 * math.pi), 1),
                "wind": round(2.5 + 1.5 * math.sin(hour / 24 * 2 * math.pi), 1),
                "humidity": 55.0,
                "pop": 10.0,
            })
    return pd.DataFrame(rows)


def get_forecast() -> Forecast:
    errors = []
    if _kma_endpoint()[0]:
        try:
            fc = Forecast(prepend_history(fetch_kma_forecast()), "기상청 단기예보")
            try:
                fc.now = fetch_kma_now()
            except Exception as e:
                fc.errors.append(f"초단기실황 실패: {e}")
            return fc
        except Exception as e:
            errors.append(f"기상청 단기예보 실패 → 대체 소스 사용: {e}")
    else:
        errors.append("기상청 API 키가 없어 Open-Meteo를 사용합니다 (.env에 KMA_API_KEY 설정).")
    try:
        return Forecast(fetch_open_meteo(), "Open-Meteo 예보", errors=errors)
    except Exception as e:
        errors.append(f"Open-Meteo 실패: {e}")
        return Forecast(demo_forecast(date.today()), "데모 데이터(오프라인)", errors=errors)


# ---------------- 안양대 캠퍼스 실측 (500m 격자) ----------------

CAMPUS_OBS = "ta,hm,ws_10m,ta_chi"


def parse_grid(text: str) -> pd.DataFrame:
    """격자 API 텍스트 응답 → time, temp, humidity, wind, feels.

    '#' 주석 줄과 'TM' 요약 줄은 건너뛰고, -999 같은 결측(아주 작은 값)이 있는 줄은 버린다.
    """
    rows = []
    for line in text.splitlines():
        parts = line.strip().replace(",", " ").split()
        if len(parts) < 5 or not parts[0].isdigit():
            continue
        try:
            vals = [float(x) for x in parts[1:5]]
        except ValueError:
            continue
        if min(vals) <= -50:
            continue
        rows.append([datetime.strptime(parts[0], "%Y%m%d%H%M"), *vals])
    return pd.DataFrame(rows, columns=["time", "temp", "humidity", "wind", "feels"])


def fetch_campus_now() -> dict:
    """안양대 위치 500m 격자의 최근 1시간 중 가장 최근 정상 값 (약 1시간 지연)."""
    key = os.environ.get("KMA_APIHUB_KEY", "").strip()
    if not key:
        raise RuntimeError("캠퍼스 실측값은 기상청 API허브 키(KMA_APIHUB_KEY)가 있어야 받을 수 있어요")
    resp = requests.get(config.CAMPUS_GRID_URL, params={
        "obs": CAMPUS_OBS, "itv": 10, "lon": config.CAMPUS_LON, "lat": config.CAMPUS_LAT,
        "help": 0, "authKey": key,
    }, timeout=(10, 30))
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}")   # URL에 키가 있으므로 상태 코드만
    df = parse_grid(resp.text)
    if df.empty:
        raise RuntimeError("최근 1시간 캠퍼스 자료가 비어 있어요")
    out = df.iloc[-1].to_dict()
    out["hour_obs"] = hour_obs(df)
    return out


def hour_obs(df: pd.DataFrame) -> dict | None:
    """격자 자료 중 가장 최근 정시(HH:00) 값 — 학습 자료(시간별)와 같은 기준.

    10분 간격 자료가 조회 시각에 맞춰 :05, :15 … 처럼 정시를 비켜 오기도 하므로,
    정시 값이 없으면 바로 앞뒤 값(정시 ±10분 안)을 시간 비율로 이어서 정시 값을 만든다.
    """
    if df.empty:
        return None
    d = df.sort_values("time").reset_index(drop=True)
    cols = ("temp", "humidity", "wind")
    for h in sorted(d["time"].dt.floor("h").unique(), reverse=True):
        h = pd.Timestamp(h)
        exact = d[d["time"] == h]
        if not exact.empty:
            r = exact.iloc[-1]
            return {"time": h, **{c: float(r[c]) for c in cols}}
        before = d[(d["time"] < h) & (d["time"] >= h - pd.Timedelta(minutes=10))]
        after = d[(d["time"] > h) & (d["time"] <= h + pd.Timedelta(minutes=10))]
        if not before.empty and not after.empty:
            a, b = before.iloc[-1], after.iloc[0]
            k = (h - a.time) / (b.time - a.time)
            return {"time": h, **{c: float(a[c] + k * (b[c] - a[c])) for c in cols}}
    return None


def nowcast_ref(obs_hour: dict | None) -> dict | None:
    """오늘 실측 반영용 기준: 정시 실측 + 그 시각 만안구 예보. 예보를 못 받으면 None."""
    if not obs_hour:
        return None
    t = pd.Timestamp(obs_hour["time"]).to_pydatetime()
    fc = None
    if getattr(config, "USE_ULTRA_NEAR", True):
        try:   # 가까운 시간은 초단기예보를 쓰므로, 오차도 그 시각에 쓰던 초단기예보 기준으로
            fc = fetch_kma_ultra_at(t)
        except Exception:
            fc = None
    if fc is None:
        fc = fetch_kma_at(t)
    return {"time": t.strftime("%Y-%m-%d %H:%M"),
            "obs": {k: obs_hour[k] for k in ("temp", "humidity", "wind")},
            "fcst": {k: fc[k] for k in ("temp", "humidity", "wind")}, "fcst_base": fc["base"]}
