"""data/의 기상청 원자료를 정리하고 안양대 격자 정답과 학습 CSV를 만든다.

원본은 수정하지 않는다.

    python prepare_training_data.py
    python prepare_training_data.py --grid-dir data/grid

grid-dir에는 grid_2022_01.csv 같은 파일이 있어야 하며 열은
tm, ta, hm, ws_10m 형식이어야 한다.
"""
from __future__ import annotations

import argparse
import json
from calendar import isleap
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_AWS = {
    2022: "22train.csv",
    2026: "26data.csv",
}
TRAIN_COLUMNS = [
    "tm", "aws_ta", "aws_hm", "aws_ws",
    "uni_ta", "uni_hm", "uni_ws", "month", "hour",
]


def read_csv_auto(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"인코딩을 읽을 수 없습니다: {path}")


def first_column(columns, prefixes):
    for prefix in prefixes:
        found = [column for column in columns if str(column).strip().startswith(prefix)]
        if found:
            return found[0]
    raise KeyError(f"열을 찾을 수 없습니다: {prefixes}; 실제 열={list(columns)}")


def clean_aws(path: Path) -> pd.DataFrame:
    raw = read_csv_auto(path)
    raw.columns = [str(column).strip() for column in raw.columns]
    if {"tm", "aws_ta", "aws_hm", "aws_ws"} <= set(raw.columns):
        out = raw[["tm", "aws_ta", "aws_hm", "aws_ws"]].copy()
    else:
        out = pd.DataFrame({
            "tm": raw[first_column(raw.columns, ["일시"])],
            "aws_ta": raw[first_column(raw.columns, ["기온"])],
            "aws_hm": raw[first_column(raw.columns, ["습도"])],
            "aws_ws": raw[first_column(raw.columns, ["풍속"])],
        })
    out["tm"] = pd.to_datetime(out["tm"], errors="coerce").dt.floor("h")
    for column in ("aws_ta", "aws_hm", "aws_ws"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out.loc[~out.aws_ta.between(-40, 50), "aws_ta"] = np.nan
    out.loc[~out.aws_hm.between(0, 100), "aws_hm"] = np.nan
    out.loc[~out.aws_ws.between(0, 60), "aws_ws"] = np.nan
    return out.dropna(subset=["tm"]).drop_duplicates("tm").sort_values("tm").reset_index(drop=True)


def clean_grid(paths: list[Path]) -> pd.DataFrame:
    parts = []
    for path in paths:
        raw = read_csv_auto(path)
        raw.columns = [str(column).strip() for column in raw.columns]
        if {"tm", "uni_ta", "uni_hm", "uni_ws"} <= set(raw.columns):
            part = raw[["tm", "uni_ta", "uni_hm", "uni_ws"]].copy()
        else:
            part = pd.DataFrame({
                "tm": raw[first_column(raw.columns, ["tm", "일시"])],
                "uni_ta": raw[first_column(raw.columns, ["ta"])],
                "uni_hm": raw[first_column(raw.columns, ["hm"])],
                "uni_ws": raw[first_column(raw.columns, ["ws_10m", "ws"])],
            })
        parts.append(part)
    grid = pd.concat(parts, ignore_index=True)
    grid["tm"] = pd.to_datetime(grid["tm"], errors="coerce").dt.floor("h")
    for column in ("uni_ta", "uni_hm", "uni_ws"):
        grid[column] = pd.to_numeric(grid[column], errors="coerce")
    grid = grid.dropna(subset=["tm", "uni_ta", "uni_hm", "uni_ws"])
    grid = grid[
        grid.uni_ta.between(-40, 50)
        & grid.uni_hm.between(0, 100)
        & grid.uni_ws.between(0, 60)
    ]
    return grid.drop_duplicates("tm").sort_values("tm").reset_index(drop=True)


def aws_quality(frame: pd.DataFrame, year: int) -> dict:
    full_hours = (366 if isleap(year) else 365) * 24
    in_year = frame[frame.tm.dt.year.eq(year)]
    present = int(in_year.tm.nunique())
    return {
        "year": year,
        "rows": len(in_year),
        "first": in_year.tm.min().isoformat() if len(in_year) else None,
        "last": in_year.tm.max().isoformat() if len(in_year) else None,
        "full_year_hours": full_hours,
        "missing_timestamps": full_hours - present,
        "coverage_percent": round(present / full_hours * 100, 2),
        "nulls": {
            column: int(in_year[column].isna().sum())
            for column in ("aws_ta", "aws_hm", "aws_ws")
        },
    }


def merge_year(aws: pd.DataFrame, grid: pd.DataFrame, year: int) -> pd.DataFrame | None:
    grid_year = grid[grid.tm.dt.year.eq(year)].copy()
    if grid_year.empty:
        return None
    # 정답 시각을 보존하고 AWS가 없는 시각은 NaN으로 둔다. 학습 로더가 짧은 공백만 보간한다.
    merged = pd.merge(grid_year, aws, on="tm", how="left")
    merged["month"] = merged.tm.dt.month
    merged["hour"] = merged.tm.dt.hour
    return merged[TRAIN_COLUMNS]


def merge_training(existing: Path, cleaned: dict[int, pd.DataFrame], grid: pd.DataFrame):
    old = pd.read_csv(existing, parse_dates=["tm"], encoding="utf-8-sig")
    old = old[TRAIN_COLUMNS]
    parts, merged_years, missing_grid_years = [old], [], []
    for year in sorted(cleaned):
        if year not in {2022, 2026}:
            continue
        merged = merge_year(cleaned[year], grid, year)
        if merged is None:
            missing_grid_years.append(year)
        else:
            parts.append(merged)
            merged_years.append(year)
    combined = pd.concat(parts, ignore_index=True)
    combined = combined[combined.tm.dt.year.between(2022, 2026)]
    combined = combined.drop_duplicates("tm", keep="last").sort_values("tm").reset_index(drop=True)
    return combined, merged_years, missing_grid_years


def main():
    parser = argparse.ArgumentParser(
        description="원본 AWS를 정리하고 2022~2026 모델 학습·평가 CSV를 준비합니다."
    )
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "data")
    parser.add_argument("--grid-dir", type=Path, default=BASE_DIR / "data" / "grid")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "data" / "prepared")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned, report = {}, {"aws": {}, "training": {}}

    for year, filename in RAW_AWS.items():
        source = data_dir / filename
        if not source.exists():
            continue
        frame = clean_aws(source)
        cleaned[year] = frame
        out = output_dir / f"aws_{year}_clean.csv"
        frame.to_csv(out, index=False, encoding="utf-8-sig", float_format="%.3f")
        report["aws"][str(year)] = aws_quality(frame, year)
        print(f"AWS {year}: {out} ({len(frame):,}줄, "
              f"커버리지 {report['aws'][str(year)]['coverage_percent']:.2f}%)")

    grid_paths = sorted(args.grid_dir.resolve().glob("grid_*.csv"))
    if grid_paths:
        grid = clean_grid(grid_paths)
        output = output_dir / "model_data.csv"
        combined, merged_years, missing_grid_years = merge_training(
            data_dir / "train_data.csv", cleaned, grid
        )
        combined.to_csv(output, index=False, encoding="utf-8-sig", float_format="%.3f")
        counts = combined.tm.dt.year.value_counts().sort_index()
        report["training"] = {
            "output": str(output),
            "rows": len(combined),
            "years": {str(year): int(count) for year, count in counts.items()},
            "merged_raw_years": merged_years,
            "missing_grid_years": missing_grid_years,
        }
        print(f"모델 데이터: {output} ({len(combined):,}줄, 원자료 병합 연도 {merged_years})")
        if missing_grid_years:
            print(f"격자 정답이 없어 병합하지 못한 연도: {missing_grid_years}")
    else:
        report["training"] = {
            "output": None,
            "reason": (
                f"안양대 격자 파일이 없습니다. {args.grid_dir.resolve()}에 "
                "grid_2022_*.csv와 평가하려는 경우 grid_2026_*.csv를 준비하세요."
            ),
        }
        print(report["training"]["reason"])

    report_path = output_dir / "quality_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"품질 리포트: {report_path}")


if __name__ == "__main__":
    main()
