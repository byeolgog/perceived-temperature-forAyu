"""안양대 캠퍼스 최신 실측값 → docs/campus_now.json (GitHub Actions가 매시간 실행).

기상청 API허브 고해상도 격자(500m) '특정지점 다중요소'로 안양대 위치의 최근 1시간 값을 받아
가장 최근 정상 값을 JSON으로 저장한다. HTML 버전(GitHub Pages)이 이 파일을 읽는다.

- 인증키는 환경변수 KMA_APIHUB_KEY에서만 읽는다 (GitHub 저장소 Secrets). 결과 파일에는 키가 들어가지 않는다.
- 받기에 실패하면 지금 공개된 campus_now.json(PREVIOUS_URL)을 그대로 다시 쓴다. 그것도 없으면 오류만 기록.

    KMA_APIHUB_KEY=키 python update_campus_now.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone

import requests

import config

OUT = config.BASE_DIR / "docs" / "campus_now.json"
KST = timezone(timedelta(hours=9))


def parse_latest(text: str) -> dict | None:
    """격자 API 텍스트 응답에서 가장 최근 정상 줄 (tm, ta, hm, ws_10m, ta_chi)."""
    latest = None
    for line in text.splitlines():
        parts = line.strip().replace(",", " ").split()
        if len(parts) < 5 or not parts[0].isdigit():
            continue          # '#' 주석, 'TM' 요약 줄
        try:
            ta, hm, ws, feels = (float(x) for x in parts[1:5])
        except ValueError:
            continue
        if min(ta, hm, ws, feels) <= -50:
            continue          # -999 결측
        t = datetime.strptime(parts[0], "%Y%m%d%H%M")
        latest = {"time": t.strftime("%Y-%m-%d %H:%M"), "temp": ta, "humidity": hm, "wind": ws, "feels": feels}
    return latest


def fetch_latest(key: str, tries: int = 3) -> dict:
    last_err = None
    for _ in range(tries):
        try:
            r = requests.get(config.CAMPUS_GRID_URL, params={
                "obs": "ta,hm,ws_10m,ta_chi", "itv": 10,
                "lon": config.CAMPUS_LON, "lat": config.CAMPUS_LAT, "help": 0, "authKey": key,
            }, timeout=(15, 45))
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")   # URL에 키가 있으므로 상태 코드만
            data = parse_latest(r.text)
            if data is None:
                raise RuntimeError("최근 1시간 자료가 비어 있음")
            return data
        except Exception as e:
            last_err = e
    raise RuntimeError(f"{tries}번 시도 모두 실패: {type(last_err).__name__} {last_err}")


def previous() -> dict | None:
    url = os.environ.get("PREVIOUS_URL")
    if not url:
        return None
    try:
        r = requests.get(url, timeout=15)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def main() -> int:
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M")
    key = os.environ.get("KMA_APIHUB_KEY", "").strip()
    try:
        if not key:
            raise RuntimeError("KMA_APIHUB_KEY 비밀값이 없음")
        data = {**fetch_latest(key), "source": "기상청 고해상도 격자(500m) 실측", "updated_at": now}
        print(f"캠퍼스 실측 {data['time']}: 기온 {data['temp']}°C, 습도 {data['humidity']}%, "
              f"풍속 {data['wind']}m/s, 체감 {data['feels']}°C")
    except Exception as e:
        print(f"실측값을 받지 못함: {e}", file=sys.stderr)
        data = previous()
        if data and "feels" in data:
            print(f"직전 값 유지 ({data.get('time')})")
            data["last_error"] = f"{now} 갱신 실패"
        else:
            data = {"error": str(e), "updated_at": now}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {OUT}")
    return 0   # 실패해도 사이트 배포는 계속 (직전 값 또는 AI 예측으로 표시)


if __name__ == "__main__":
    sys.exit(main())
