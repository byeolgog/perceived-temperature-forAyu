"""실시간 날씨 파일 만들기 (GitHub Actions가 매시간 실행 → GitHub Pages와 함께 배포).

  docs/campus_now.json : 안양대 500m 격자 최신 실측 (기온·습도·풍속·체감온도)
                         + nowcast: 정시 실측과 그 시각 만안구 예보 (오늘 실측 반영용)
  docs/forecast.json   : 기상청 만안구 초단기실황(지금 기온) + 단기예보(시간별, 약 3일)
  docs/history.json    : 오늘·어제 시간별 캠퍼스 실측 + 그 시각을 사이트가 어떻게 예측했는지
                         (처음 예측 / 직전 예측) — 지난 시간의 '예측 vs 실측' 비교용

HTML 버전(GitHub Pages)이 이 파일들을 읽어, 인증키 없이도 기상청 값으로 화면을 그린다.

- 인증키는 환경변수 KMA_APIHUB_KEY에서만 읽는다 (GitHub 저장소 Secrets). 결과 파일에는 키가 없다.
- 받기에 실패하면 지금 공개된 파일(PAGES_URL/파일이름)을 그대로 다시 쓴다. 그것도 없으면 오류만 기록.

    KMA_APIHUB_KEY=키 python update_live_weather.py
"""
from __future__ import annotations

import json
import os
import sys

import requests

import config
import site_model
import weather

DOCS = config.BASE_DIR / "docs"


def parse_latest(text: str) -> dict | None:
    """격자 API 텍스트 응답에서 가장 최근 정상 줄 (tm, ta, hm, ws_10m, ta_chi) + 최근 정시 값."""
    df = weather.parse_grid(text)
    if df.empty:
        return None
    r = df.iloc[-1]
    return {"time": r.time.strftime("%Y-%m-%d %H:%M"), "temp": float(r.temp),
            "humidity": float(r.humidity), "wind": float(r.wind), "feels": float(r.feels),
            "_hour_obs": weather.hour_obs(df)}


def add_nowcast(data: dict) -> dict:
    """정시 실측 시각의 만안구 예보를 붙인다. 실패해도 실측값은 그대로 (보정만 생략)."""
    obs = data.pop("_hour_obs", None)
    for _ in range(3):
        try:
            ref = weather.nowcast_ref(obs)
            if ref:
                data["nowcast"] = ref
            else:
                data["nowcast_error"] = "최근 1시간에 정시 실측이 없음"
            return data
        except Exception as e:
            last = e
    data["nowcast_error"] = f"그 시각 예보를 받지 못함: {last}"
    return data


def fetch_campus(key: str, tries: int = 3) -> dict:
    last = None
    for _ in range(tries):
        try:
            r = requests.get(config.CAMPUS_GRID_URL, params={
                "obs": weather.CAMPUS_OBS, "itv": 10,
                "lon": config.CAMPUS_LON, "lat": config.CAMPUS_LAT, "help": 0, "authKey": key,
            }, timeout=(15, 45))
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")   # URL에 키가 있으므로 상태 코드만
            data = parse_latest(r.text)
            if data is None:
                raise RuntimeError("최근 1시간 자료가 비어 있음")
            return add_nowcast({**data, "source": "기상청 고해상도 격자(500m) 실측"})
        except Exception as e:
            last = e
    raise RuntimeError(f"{tries}번 시도 모두 실패: {type(last).__name__} {last}")


def fetch_forecast(tries: int = 3) -> dict:
    last = None
    for _ in range(tries):
        try:
            df = weather.fetch_kma_forecast()
            break
        except Exception as e:
            last = e
    else:
        raise RuntimeError(f"단기예보 {tries}번 시도 모두 실패: {last}")
    out = {
        "source": "기상청 단기예보",
        "area": config.APP_REF_NAME,
        "hourly": [{"date": t.strftime("%Y-%m-%d"), "hour": t.hour, "temp": r.temp, "wind": r.wind,
                    "humidity": r.humidity, "pop": r.pop}
                   for t, r in zip(df["time"], df.itertuples(index=False))],
    }
    try:
        now = weather.fetch_kma_now()
        out["now"] = {"time": now["time"].strftime("%Y-%m-%d %H:%M"), "temp": now["temp"],
                      "wind": now["wind"], "humidity": now["humidity"], "source": "기상청 초단기실황"}
    except Exception as e:
        out["now_error"] = str(e)
    return out


def previous(name: str) -> dict | None:
    base = os.environ.get("PAGES_URL")
    if not base:
        return None
    for _ in range(3):   # 기록 파일은 이어 쓰므로 한 번 실패로 날아가지 않게 재시도
        try:
            r = requests.get(f"{base.rstrip('/')}/{name}", timeout=15)
            if r.status_code == 404:
                return None
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
    return None


def write(name: str, make, ok_key: str, stamp: str) -> None:
    """make()로 새 값을 받고, 실패하면 공개된 직전 파일을 유지."""
    try:
        data = {**make(), "updated_at": stamp}
        print(f"{name}: 갱신 성공")
    except Exception as e:
        print(f"{name}: 받지 못함 — {e}", file=sys.stderr)
        data = previous(name)
        if data and ok_key in data:
            data["last_error"] = f"{stamp} 갱신 실패"
            print(f"{name}: 직전 값 유지 (갱신 {data.get('updated_at')})")
        else:
            data = {"error": str(e), "updated_at": stamp}
    DOCS.mkdir(exist_ok=True)
    (DOCS / name).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def update_history(stamp: str) -> None:
    """history.json: 직전 공개본을 이어받아 이번 시간의 실측과 예측을 더한다 (어제·오늘만 보관)."""
    from datetime import datetime, timedelta
    now = datetime.strptime(stamp, "%Y-%m-%d %H:%M")
    hist = previous("history.json") or {}
    obs, pred = hist.get("obs", {}), hist.get("pred", {})

    campus = json.loads((DOCS / "campus_now.json").read_text(encoding="utf-8"))
    fc = json.loads((DOCS / "forecast.json").read_text(encoding="utf-8"))

    nc = campus.get("nowcast")     # 사이트가 지금 쓰는 실측 (이번에 못 받았으면 직전 값 — 그래도 실제 관측)
    if nc:
        t = datetime.strptime(nc["time"], "%Y-%m-%d %H:%M")
        o = nc["obs"]
        obs[nc["time"]] = {"temp": round(o["temp"], 1), "humidity": round(o["humidity"]),
                           "wind": round(o["wind"], 1),
                           "feels": round(site_model.feels(t.month, o["temp"], o["humidity"], o["wind"]), 1)}

    if fc.get("hourly"):   # 사이트가 지금 보여주는 예측 (민감도 보통, 사이트와 같은 파일·같은 계산)
        err = site_model.nowcast_error(nc) if nc else None
        for r in fc["hourly"]:
            t = datetime.strptime(f"{r['date']} {r['hour']:02d}:00", "%Y-%m-%d %H:%M")
            if t <= now or t.date() != now.date():
                continue                               # 아직 오지 않은 오늘 시각만
            f = round(site_model.campus_feels(r["temp"], r["wind"], r["humidity"], t, err)["feels"], 1)
            key = t.strftime("%Y-%m-%d %H:%M")
            entry = pred.setdefault(key, {"first": {"feels": f, "made": stamp}})
            entry["last"] = {"feels": f, "made": stamp}

    keep = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    data = {"obs": {k: v for k, v in sorted(obs.items()) if k >= keep},
            "pred": {k: v for k, v in sorted(pred.items()) if k >= keep},
            "updated_at": stamp}
    (DOCS / "history.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    both = [k for k in data["obs"] if k in data["pred"] and k.startswith(now.strftime("%Y-%m-%d"))]
    print(f"history.json: 실측 {len(data['obs'])}시간, 예측 {len(data['pred'])}시간, 오늘 비교 가능 {len(both)}시간")


def main() -> int:
    stamp = weather.kst_now().strftime("%Y-%m-%d %H:%M")
    key = os.environ.get("KMA_APIHUB_KEY", "").strip()

    def campus():
        if not key:
            raise RuntimeError("KMA_APIHUB_KEY 비밀값이 없음")
        return fetch_campus(key)

    def forecast():
        if not key and not os.environ.get("KMA_API_KEY"):
            raise RuntimeError("KMA_APIHUB_KEY 비밀값이 없음")
        return fetch_forecast()

    write("campus_now.json", campus, "feels", stamp)
    write("forecast.json", forecast, "hourly", stamp)
    try:
        update_history(stamp)
    except Exception as e:   # 기록이 실패해도 사이트 배포는 계속
        print(f"history.json: 기록 실패 — {type(e).__name__} {e}", file=sys.stderr)

    for name in ("campus_now.json", "forecast.json"):
        js = json.loads((DOCS / name).read_text(encoding="utf-8"))
        if "feels" in js:
            print(f"  캠퍼스 실측 {js['time']}: 기온 {js['temp']}°C, 습도 {js['humidity']}%, "
                  f"풍속 {js['wind']}m/s, 체감 {js['feels']}°C")
            nc = js.get("nowcast")
            if nc:
                print(f"  실측 반영 기준 {nc['time']}: 실측 {nc['obs']} / 예보({nc['fcst_base']} 발표) {nc['fcst']}")
            else:
                print(f"  실측 반영 없음: {js.get('nowcast_error')}")
        if "hourly" in js:
            now = js.get("now")
            print(f"  만안구 예보 {len(js['hourly'])}시간"
                  + (f", 지금 {now['time']} 기온 {now['temp']}°C" if now else f", 실황 실패: {js.get('now_error')}"))
    return 0   # 실패해도 사이트 배포는 계속 (직전 값 또는 Open-Meteo·AI 예측으로 표시)


if __name__ == "__main__":
    sys.exit(main())
