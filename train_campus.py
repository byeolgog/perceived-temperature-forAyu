"""안양대 캠퍼스 예측 모델 학습.

    python train_campus.py                       # data/train_data.csv → data/campus_model.joblib
    python train_campus.py --data 경로.csv --test-year 2025

1) 시험: test-year 이전 해로 학습하고 test-year로 채점 (적중률 출력)
2) 저장: 전체 데이터로 다시 학습해서 campus_model.joblib 저장
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from campus_model import BASES, TARGETS, CampusModel, kma_feels

BASE_DIR = Path(__file__).resolve().parent


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["tm"], encoding="utf-8-sig")
    need = ["tm", *BASES, *TARGETS, "month", "hour"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"train_data.csv에 없는 열: {missing}")
    return df[need].dropna().reset_index(drop=True)


def evaluate(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    model = CampusModel().fit(train)
    pred = model.predict(test)
    real = kma_feels(test.month, test.uni_ta, test.uni_hm, test.uni_ws)
    rows = {}
    for name, p in {"보정 안 함(날씨앱)": test[BASES].to_numpy(dtype=float), "랜덤포레스트": pred}.items():
        err = np.abs(kma_feels(test.month, p[:, 0], p[:, 1], p[:, 2]) - real)
        rows[name] = {
            "기온 오차(°C)": np.abs(p[:, 0] - test.uni_ta).mean(),
            "습도 오차(%)": np.abs(p[:, 1] - test.uni_hm).mean(),
            "풍속 오차(m/s)": np.abs(p[:, 2] - test.uni_ws).mean(),
            "체감 오차(°C)": err.mean(),
            "±1°C 적중(%)": (err <= 1).mean() * 100,
            "±2°C 적중(%)": (err <= 2).mean() * 100,
        }
    return pd.DataFrame(rows).T.round(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=BASE_DIR / "data" / "train_data.csv", type=Path)
    ap.add_argument("--out", default=BASE_DIR / "data" / "campus_model.joblib", type=Path)
    ap.add_argument("--test-year", default=2025, type=int)
    args = ap.parse_args()

    data = load_data(args.data)
    years = sorted(data.tm.dt.year.unique())
    print(f"데이터 {len(data)}줄 ({data.tm.min():%Y-%m-%d} ~ {data.tm.max():%Y-%m-%d}), 연도 {years}")

    train = data[data.tm.dt.year < args.test_year]
    test = data[data.tm.dt.year == args.test_year]
    if len(train) and len(test):
        print(f"\n[시험] 학습 {len(train)}줄 / 시험 {args.test_year}년 {len(test)}줄")
        print(evaluate(train, test).to_string())
    else:
        print(f"\n[시험] {args.test_year}년 앞뒤 데이터가 없어 채점은 건너뜀")

    t0 = time.time()
    CampusModel().fit(data).save(args.out)
    size = args.out.stat().st_size / 1e6
    print(f"\n[저장] 전체 {len(data)}줄로 학습 → {args.out} ({size:.1f} MB, {time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
