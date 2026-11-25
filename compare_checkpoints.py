"""2024년 검증 1등 체크포인트를 고정한 뒤 2025년 테스트로 최종 평가한다.

    python compare_checkpoints.py --test-year 2025
    python compare_checkpoints.py --models rf lightgbm lstm --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from campus_model import BASES, CampusModel, tidy
from compare_models import (
    BASE_DIR,
    REF_MODEL,
    SEQ_FEATURES,
    TRAIN_MODEL_NAMES,
    TRAIN_MODEL_ORDER,
    build_nets,
    lag_features,
    load_hourly,
    prediction_frame,
    print_report,
    resolve_torch_device,
    result_frame,
    score,
    season_frame,
    windows,
    write_json,
)


def read_metadata(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"학습 기록이 없습니다: {path}") from exc


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def comparison_signature(metadata: dict) -> tuple:
    settings = metadata["settings"]
    rows = metadata["rows"]
    try:
        return (
            settings["train_val_sha256"],
            settings["train_start_year"],
            settings["val_year"],
            settings["window"],
            rows["train"],
            rows["validation"],
            metadata["overall_score"]["version"],
        )
    except KeyError as exc:
        raise SystemExit(
            "이 체크포인트는 train/validation 분리 이전 형식입니다. train_model.py로 다시 학습하세요."
        ) from exc


def load_test_data(path: Path, test_year: int, window: int):
    df = load_hourly(path)
    lag = lag_features(df)
    year = df.tm.dt.year.to_numpy()
    need = max(window, 25)
    seq_ok = df[BASES].notna().all(axis=1).rolling(need).sum().eq(need).to_numpy()
    te_rows = np.where(df.observed.to_numpy() & seq_ok & (year == test_year))[0]
    test = df.iloc[te_rows]
    if not len(test):
        years = sorted(df.loc[df.observed, "tm"].dt.year.unique().tolist())
        raise SystemExit(
            f"{test_year}년 테스트 데이터가 없습니다: {path}\n"
            f"현재 포함 연도: {years}. 요청한 테스트 연도가 포함된 CSV를 --data로 지정하세요."
        )
    return df, lag, te_rows, test


def load_torch_checkpoint(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch 2.5 이하
        return torch.load(path, map_location="cpu")


def predict_checkpoint(key: str, path: Path, df: pd.DataFrame, lag: pd.DataFrame,
                       te_rows: np.ndarray, test: pd.DataFrame, device: str) -> np.ndarray:
    base = test[BASES].to_numpy(float)
    if key == "rf":
        return CampusModel.load(path).predict(test)

    if key in {"rf-lag", "lightgbm"}:
        saved = joblib.load(path)
        features = saved["features"]
        delta = np.column_stack([
            model.predict(lag.iloc[te_rows][features]) for model in saved["models"]
        ])
        return tidy(base + delta)

    import torch

    saved = load_torch_checkpoint(path)
    LSTMNet, TSMixer = build_nets()
    channels = len(SEQ_FEATURES)
    make = (
        (lambda: LSTMNet(channels))
        if key == "lstm"
        else (lambda: TSMixer(saved["window"], channels))
    )
    net = make().to(device)
    net.load_state_dict(saved["state_dict"])
    net.eval()
    sequence = windows(
        df,
        te_rows,
        saved["window"],
        np.asarray(saved["input_mean"]),
        np.asarray(saved["input_std"]),
    )
    with torch.no_grad():
        scaled = net(torch.from_numpy(sequence).to(device)).cpu().numpy()
    delta = scaled * np.asarray(saved["target_std"]) + np.asarray(saved["target_mean"])
    return tidy(base + delta)


def main():
    parser = argparse.ArgumentParser(
        description="검증 1등 체크포인트를 선택한 뒤 별도의 테스트 연도로 최종 평가합니다."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=BASE_DIR / "compare_runs" / "checkpoints",
        help="train_model.py에서 공통으로 사용한 실험 폴더",
    )
    parser.add_argument("--data", type=Path, default=BASE_DIR / "data" / "train_data.csv")
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--select-only",
        action="store_true",
        help="2024 검증 결과로 최종 모델만 선택하고 테스트는 실행하지 않음",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["all", *TRAIN_MODEL_ORDER],
        default=["all"],
        metavar="MODEL",
    )
    args = parser.parse_args()
    if "all" in args.models and len(args.models) > 1:
        parser.error("--models all은 다른 모델 이름과 함께 쓸 수 없습니다.")

    experiment_dir = args.experiment_dir.resolve()
    requested = list(TRAIN_MODEL_ORDER if args.models == ["all"] else args.models)
    available = [
        key for key in TRAIN_MODEL_ORDER
        if key in requested and (experiment_dir / key / "metadata.json").exists()
    ]
    missing = [key for key in requested if key not in available]
    if missing and args.models != ["all"]:
        raise SystemExit("먼저 학습해야 하는 모델: " + ", ".join(missing))
    if not available:
        raise SystemExit(f"완료된 모델 체크포인트가 없습니다: {experiment_dir}")

    metadata_by_model = {}
    signature = None
    for key in available:
        metadata = read_metadata(experiment_dir / key / "metadata.json")
        if metadata.get("status") != "completed":
            raise SystemExit(f"{key} 학습이 완료되지 않았습니다: {metadata.get('status')}")
        current_signature = comparison_signature(metadata)
        if signature is None:
            signature = current_signature
        elif current_signature != signature:
            raise SystemExit(
                f"{key}의 학습/검증 설정이 다른 모델과 다릅니다. "
                "같은 --data, --train-start-year, --val-year, --window로 다시 학습하세요."
            )
        checkpoint = experiment_dir / key / metadata["checkpoint"]
        if not checkpoint.exists():
            raise SystemExit(f"{key} 체크포인트 파일이 없습니다: {checkpoint}")
        expected_hash = metadata.get("checkpoint_sha256")
        if expected_hash and file_sha256(checkpoint) != expected_hash:
            raise SystemExit(f"{key} 체크포인트가 학습 당시 파일과 다릅니다: {checkpoint}")
        metadata_by_model[key] = (metadata, checkpoint)

    train_start_year, val_year, window = signature[1], signature[2], signature[3]
    if args.test_year <= val_year:
        parser.error(f"--test-year는 검증 연도 {val_year}보다 뒤여야 합니다.")

    # 검증 연도(기본 2024) 결과만으로 최종 모델 하나를 먼저 고정한다.
    validation_results = {}
    validation_info, validation_timings, validation_checkpoints = {}, {}, {}
    validation_baseline = None
    validation_seasons, validation_season_baseline = {}, None
    for key in available:
        metadata, checkpoint = metadata_by_model[key]
        name = TRAIN_MODEL_NAMES[key]
        current_baseline = metadata["validation_baseline_metrics"]
        if validation_baseline is None:
            validation_baseline = current_baseline
        elif not all(
            np.isclose(validation_baseline[metric], current_baseline[metric])
            for metric in validation_baseline
        ):
            raise SystemExit(f"{key}의 검증 기준선이 달라 공정하게 선택할 수 없습니다.")
        validation_results[name] = metadata["validation_metrics"]
        validation_timings[name] = float(metadata["training_seconds"])
        validation_checkpoints[name] = checkpoint.name
        validation_info[name] = checkpoint.name
        season_path = experiment_dir / key / "validation_season_metrics.csv"
        if not season_path.is_file():
            raise SystemExit(f"계절별 검증 결과가 없습니다: {season_path}")
        current_seasons = pd.read_csv(season_path, encoding="utf-8-sig", index_col="model")
        current_season_baseline = current_seasons.loc["보정 안 함(날씨앱)"]
        if validation_season_baseline is None:
            validation_season_baseline = current_season_baseline
        elif not np.allclose(validation_season_baseline, current_season_baseline):
            raise SystemExit(f"{key}의 계절별 검증 기준선이 달라 공정하게 비교할 수 없습니다.")
        validation_seasons[name] = current_seasons.loc[name].to_dict()

    ordered_validation = {"보정 안 함(날씨앱)": validation_baseline}
    ordered_validation.update(validation_results)
    validation_ref = REF_MODEL if "rf" in available else None
    print_report(
        ordered_validation,
        validation_info,
        ref=validation_ref,
        title=(
            f"모델 선택: 학습 {train_start_year}~{val_year - 1}년 · "
            f"검증 {val_year}년"
        ),
    )
    validation_frame = result_frame(
        ordered_validation, validation_info, validation_timings, ref=validation_ref
    )
    validation_frame["체크포인트"] = (
        pd.Series(validation_checkpoints).reindex(validation_frame.index).fillna("-")
    )
    validation_frame.to_csv(
        experiment_dir / "validation_comparison.csv",
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    validation_season_frame = pd.DataFrame.from_dict(
        {"보정 안 함(날씨앱)": validation_season_baseline.to_dict(), **validation_seasons},
        orient="index",
    ).reindex(columns=["봄", "여름", "가을", "겨울"])
    validation_season_frame.index.name = "model"
    validation_season_frame.to_csv(
        experiment_dir / "validation_season_comparison.csv",
        encoding="utf-8-sig",
        float_format="%.4f",
    )

    best_key = max(
        available,
        key=lambda key: float(metadata_by_model[key][0]["validation_metrics"]["종합 점수"]),
    )
    best_metadata, best_checkpoint = metadata_by_model[best_key]
    best_name = TRAIN_MODEL_NAMES[best_key]
    best_validation_score = float(best_metadata["validation_metrics"]["종합 점수"])
    write_json(
        experiment_dir / "selection.json",
        {
            "selected_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "selected_model": best_key,
            "selected_model_name": best_name,
            "checkpoint": str(best_checkpoint.relative_to(experiment_dir)),
            "checkpoint_sha256": best_metadata.get("checkpoint_sha256"),
            "validation_year": val_year,
            "validation_overall_score": best_validation_score,
            "candidate_models": available,
            "score_version": signature[6],
        },
    )
    print(f"\n검증 기준 최종 선택: {best_name} ({best_validation_score:.1f}점)")
    if missing:
        print("미학습 모델(선택 후보에서 제외): " + ", ".join(missing))
    if args.select_only:
        print(f"선택 저장: {experiment_dir / 'selection.json'}")
        return

    # 선택을 끝낸 뒤에만 테스트 데이터를 읽는다. 모든 후보를 평가하되 선택은 바꾸지 않는다.
    df, lag, te_rows, test = load_test_data(args.data.resolve(), args.test_year, window)
    evaluation_keys = available
    has_neural_model = any(key in {"lstm", "tsmixer"} for key in evaluation_keys)
    device = resolve_torch_device(args.device) if has_neural_model else "cpu"
    if has_neural_model:
        print(f"신경망 추론 장치: {device}")

    base = test[BASES].to_numpy(float)
    results = {"보정 안 함(날씨앱)": score(test, base)}
    predictions = {"보정 안 함(날씨앱)": base}
    info, timings, checkpoints, validation_scores = {}, {}, {}, {}
    for key in evaluation_keys:
        metadata, checkpoint = metadata_by_model[key]
        name = TRAIN_MODEL_NAMES[key]
        prediction = predict_checkpoint(key, checkpoint, df, lag, te_rows, test, device)
        predictions[name] = prediction
        results[name] = score(test, prediction)
        elapsed = float(metadata["training_seconds"])
        timings[name] = elapsed
        val_score = float(metadata["validation_metrics"]["종합 점수"])
        validation_scores[name] = val_score
        selected_mark = "2024 선택 · " if key == best_key else ""
        info[name] = f"{selected_mark}검증 {val_score:.1f} · {checkpoint.name}"
        checkpoints[name] = checkpoint.name

    ref = REF_MODEL if "rf" in evaluation_keys else None
    title = (
        f"전체 후보 테스트 비교 (2024 선택 모델: {best_name}) · "
        f"학습 {train_start_year}~{val_year - 1}년 · "
        f"검증 {val_year}년 · 시험 {test.tm.min():%Y-%m-%d}~{test.tm.max():%Y-%m-%d} "
        f"{len(test):,}시간"
    )
    print_report(results, info, ref=ref, title=title)

    frame = result_frame(results, info, timings, ref=ref)
    frame["검증 종합 점수"] = pd.Series(validation_scores).reindex(frame.index)
    frame["2024 검증 선택"] = [name == best_name for name in frame.index]
    frame["체크포인트"] = pd.Series(checkpoints).reindex(frame.index).fillna("-")
    comparison_path = experiment_dir / f"comparison_{args.test_year}.csv"
    frame.to_csv(comparison_path, encoding="utf-8-sig", float_format="%.6f")
    season_frame(test, predictions).to_csv(
        experiment_dir / f"comparison_{args.test_year}_season_metrics.csv",
        encoding="utf-8-sig",
        float_format="%.4f",
    )
    prediction_frame(test, predictions).to_csv(
        experiment_dir / f"comparison_{args.test_year}_predictions.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.4f",
    )
    write_json(
        experiment_dir / f"comparison_{args.test_year}.json",
        {
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "candidate_models": available,
            "selected_model": best_key,
            "tested_models": evaluation_keys,
            "missing_models": missing,
            "train_val_sha256": signature[0],
            "test_data_sha256": file_sha256(args.data.resolve()),
            "train_start_year": train_start_year,
            "val_year": val_year,
            "test_year": args.test_year,
            "test_first": test.tm.min().isoformat(),
            "test_last": test.tm.max().isoformat(),
            "window": window,
            "train_rows": signature[4],
            "validation_rows": signature[5],
            "test_rows": len(test),
            "score_version": signature[6],
            "device": device,
        },
    )
    print(f"\n비교 저장: {comparison_path}")


if __name__ == "__main__":
    main()
