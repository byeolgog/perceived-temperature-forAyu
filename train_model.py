"""비교 모델 하나를 학습하고 최선의 체크포인트와 시험 예측을 저장한다.

예:
    python train_model.py --model rf
    python train_model.py --model lightgbm
    python train_model.py --model lstm --seeds 3 --device cuda
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from campus_model import BASES, FEATURES, RF_PARAMS, TARGETS, CampusModel, add_features, tidy
from compare_models import (
    BASE_DIR,
    OVERALL_SCORE_FORMULA,
    OVERALL_SCORE_VERSION,
    REF_MODEL,
    SEQ_FEATURES,
    TRAIN_MODEL_NAMES,
    Tee,
    build_nets,
    lag_features,
    load_hourly,
    package_versions,
    prediction_frame,
    resolve_torch_device,
    result_frame,
    score,
    season_frame,
    train_net,
    windows,
    write_json,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_val_sha256(path: Path, train_start_year: int, val_year: int) -> str:
    """시험 행과 무관하게 지정한 학습·검증 기간만으로 해시를 만든다."""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    tm = pd.to_datetime(frame["tm"])
    frame = frame.loc[tm.dt.year.between(train_start_year, val_year)].copy()
    frame["_tm_sort"] = pd.to_datetime(frame["tm"])
    frame = frame.sort_values("_tm_sort").drop(columns="_tm_sort").reset_index(drop=True)
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_metrics(values: dict) -> dict:
    return {key: float(value) for key, value in values.items()}


def fit_rf_with_progress(
    x_train,
    y_train: np.ndarray,
    x_val,
    base_val: np.ndarray,
    val: pd.DataFrame,
    params: dict,
    label: str,
):
    """랜덤포레스트를 누적 학습하며 중간 검증 성능을 로그에 남긴다."""
    from sklearn.ensemble import RandomForestRegressor

    total_trees = int(params["n_estimators"])
    stages = list(range(min(25, total_trees), total_trees + 1, 25))
    if not stages or stages[-1] != total_trees:
        stages.append(total_trees)
    model_params = {**params, "n_estimators": stages[0], "warm_start": True}
    models = [RandomForestRegressor(**model_params) for _ in TARGETS]
    prediction = None
    for trees in stages:
        stage_started = time.time()
        for target_index, model in enumerate(models):
            model.set_params(n_estimators=trees)
            model.fit(x_train, y_train[:, target_index])
        prediction = tidy(
            base_val
            + np.column_stack([model.predict(x_val) for model in models])
        )
        metrics = score(val, prediction)
        print(
            f"{label} 진행  트리 {trees:>3}/{total_trees} | "
            f"종합 {metrics['종합 점수']:.1f} | "
            f"체감 MAE {metrics['체감 오차(°C)']:.3f}°C | "
            f"±1°C {metrics['±1°C 적중(%)']:.1f}% | "
            f"단계 {time.time() - stage_started:.1f}초",
            flush=True,
        )
    return models, prediction


def prepare_data(args):
    df = load_hourly(args.data)
    lag = lag_features(df)
    year = df.tm.dt.year.to_numpy()
    need = max(args.window, 25)
    seq_ok = df[BASES].notna().all(axis=1).rolling(need).sum().eq(need).to_numpy()
    usable = df.observed.to_numpy() & seq_ok
    tr_rows = np.where(usable & (year >= args.train_start_year) & (year < args.val_year))[0]
    va_rows = np.where(usable & (year == args.val_year))[0]
    train, val = df.iloc[tr_rows], df.iloc[va_rows]
    years = sorted(df.loc[df.observed, "tm"].dt.year.unique().tolist())
    required_train_years = set(range(args.train_start_year, args.val_year))
    missing_train_years = sorted(required_train_years - set(years))
    if missing_train_years or not len(train) or not len(val):
        raise ValueError(
            f"학습({args.train_start_year}~{args.val_year - 1}: {len(train):,}줄)과 "
            f"검증({args.val_year}: {len(val):,}줄)이 모두 필요합니다. "
            f"누락 학습 연도: {missing_train_years or '없음'}, 현재 연도: {years}"
        )
    base_va = val[BASES].to_numpy(float)
    delta_train = train[TARGETS].to_numpy(float) - train[BASES].to_numpy(float)
    delta_val = val[TARGETS].to_numpy(float) - val[BASES].to_numpy(float)
    return df, lag, tr_rows, va_rows, train, val, base_va, delta_train, delta_val


def train_one(args, model_dir: Path) -> tuple[dict, str]:
    (df, lag, tr_rows, va_rows, train, val,
     base_va, delta_train, delta_val) = prepare_data(args)
    name = TRAIN_MODEL_NAMES[args.model]
    print(f"데이터     학습 {len(train):,}시간 ({args.train_start_year}~{args.val_year - 1}) / "
          f"검증 {len(val):,}시간 ({args.val_year})")
    print(f"모델       {name}")
    started = time.time()
    validation_loss = None
    selected_seed = None
    epochs = None
    n_params = None
    device = "cpu"

    if args.model == "rf":
        train_features = add_features(train)[FEATURES]
        val_features = add_features(val)[FEATURES]
        models, validation_prediction = fit_rf_with_progress(
            train_features,
            delta_train,
            val_features,
            base_va,
            val,
            RF_PARAMS,
            "RF",
        )
        model = CampusModel()
        model.models = dict(zip(TARGETS, models))
        checkpoint = model_dir / "best.joblib"
        model.save(checkpoint)

    elif args.model == "rf-lag":
        models, validation_prediction = fit_rf_with_progress(
            lag.iloc[tr_rows],
            delta_train,
            lag.iloc[va_rows],
            base_va,
            val,
            RF_PARAMS,
            "RF-lag",
        )
        checkpoint = model_dir / "best.joblib"
        joblib.dump({"models": models, "features": list(lag.columns)}, checkpoint, compress=3)

    elif args.model == "lightgbm":
        import lightgbm as lgb

        models, val_parts = [], []
        for target_index in range(3):
            print(f"LightGBM 대상 {target_index + 1}/3 ({TARGETS[target_index]})", flush=True)
            model = lgb.LGBMRegressor(
                n_estimators=600,
                learning_rate=0.03,
                num_leaves=31,
                min_child_samples=40,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective="huber",
                alpha=1.0,
                verbose=-1,
                random_state=0,
                n_jobs=-1,
            ).fit(
                lag.iloc[tr_rows],
                delta_train[:, target_index],
                eval_set=[(lag.iloc[va_rows], delta_val[:, target_index])],
                eval_metric="l1",
                callbacks=[lgb.log_evaluation(period=50)],
            )
            models.append(model)
            target_prediction = model.predict(lag.iloc[va_rows])
            val_parts.append(target_prediction)
            target_mae = np.abs(target_prediction - delta_val[:, target_index]).mean()
            print(f"LightGBM 대상 완료 | 보정량 검증 MAE {target_mae:.4f}", flush=True)
        validation_prediction = tidy(base_va + np.column_stack(val_parts))
        checkpoint = model_dir / "best.joblib"
        joblib.dump({"models": models, "features": list(lag.columns)}, checkpoint, compress=3)

    else:
        import torch

        device = resolve_torch_device(args.device)
        print(f"장치       {device}" + (
            f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""
        ))
        LSTMNet, TSMixer = build_nets()
        features = df[SEQ_FEATURES].to_numpy(np.float32)
        mu = np.nanmean(features[tr_rows], axis=0)
        sd = np.nanstd(features[tr_rows], axis=0) + 1e-6
        Xtr = windows(df, tr_rows, args.window, mu, sd)
        Xval = windows(df, va_rows, args.window, mu, sd)
        ymu = delta_train.mean(axis=0)
        ysd = delta_train.std(axis=0) + 1e-6
        Ytr = ((delta_train - ymu) / ysd).astype(np.float32)
        Yval = ((delta_val - ymu) / ysd).astype(np.float32)
        channels = len(SEQ_FEATURES)
        make = (
            (lambda: LSTMNet(channels))
            if args.model == "lstm"
            else (lambda: TSMixer(args.window, channels))
        )

        best_run = None
        for seed in range(args.seeds):
            seed_started = time.time()

            def epoch_progress(epoch, total, val_loss, best_loss, bad_epochs, learning_rate):
                print(
                    f"시드 {seed + 1}/{args.seeds} | 에폭 {epoch:>2}/{total} | "
                    f"검증 loss {val_loss:.5f} | best {best_loss:.5f} | "
                    f"대기 {bad_epochs}/{args.patience} | lr {learning_rate:.6f}",
                    flush=True,
                )

            val_scaled, params, seed_epochs, seed_val, state = train_net(
                make,
                Xtr,
                Ytr,
                Xval,
                seed=seed,
                device=device,
                epochs=args.epochs,
                patience=args.patience,
                Xval=Xval,
                Yval=Yval,
                progress=epoch_progress,
            )
            print(f"시드 {seed + 1}/{args.seeds}  검증 {seed_val:.5f} / "
                  f"{seed_epochs}에폭 / {time.time() - seed_started:.0f}초")
            if best_run is None or seed_val < best_run["validation_loss"]:
                best_run = {
                    "validation_prediction": val_scaled,
                    "state_dict": state,
                    "validation_loss": seed_val,
                    "seed": seed,
                    "epochs": seed_epochs,
                    "n_params": params,
                }

        validation_loss = float(best_run["validation_loss"])
        selected_seed = int(best_run["seed"])
        epochs = int(best_run["epochs"])
        n_params = int(best_run["n_params"])
        validation_prediction = tidy(base_va + best_run["validation_prediction"] * ysd + ymu)
        checkpoint = model_dir / "best.pt"
        torch.save(
            {
                "model": args.model,
                "state_dict": best_run["state_dict"],
                "window": args.window,
                "sequence_features": SEQ_FEATURES,
                "input_mean": mu,
                "input_std": sd,
                "target_mean": ymu,
                "target_std": ysd,
                "seed": selected_seed,
                "validation_loss": validation_loss,
                "epochs": epochs,
                "n_params": n_params,
            },
            checkpoint,
        )

    elapsed = time.time() - started
    validation_score = score(val, validation_prediction)
    validation_baseline_score = score(val, base_va)
    info = f"{elapsed:.0f}초"
    timings = {name: elapsed}
    validation_summary = result_frame(
        {"보정 안 함(날씨앱)": validation_baseline_score, name: validation_score},
        {name: info},
        timings,
        ref=REF_MODEL if args.model == "rf" else None,
    )
    validation_summary.to_csv(
        model_dir / "validation_metrics.csv", encoding="utf-8-sig", float_format="%.4f"
    )
    season_frame(val, {"보정 안 함(날씨앱)": base_va, name: validation_prediction}).to_csv(
        model_dir / "validation_season_metrics.csv", encoding="utf-8-sig", float_format="%.4f"
    )
    prediction_frame(val, {"보정 안 함(날씨앱)": base_va, name: validation_prediction}).to_csv(
        model_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig", float_format="%.4f"
    )
    print(f"검증 점수  {validation_score['종합 점수']:.1f}")
    print(f"체크포인트 {checkpoint}")

    details = {
        "model": args.model,
        "model_name": name,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": file_sha256(checkpoint),
        "training_seconds": elapsed,
        "validation_loss": validation_loss,
        "selected_seed": selected_seed,
        "epochs": epochs,
        "n_params": n_params,
        "device": device,
        "rows": {"train": len(train), "validation": len(val)},
        "validation_metrics": json_metrics(validation_score),
        "validation_baseline_metrics": json_metrics(validation_baseline_score),
    }
    return details, checkpoint.name


def main():
    parser = argparse.ArgumentParser(description="비교 모델 하나를 학습하고 최선의 체크포인트만 저장합니다.")
    parser.add_argument("--model", required=True, choices=list(TRAIN_MODEL_NAMES))
    parser.add_argument("--data", type=Path, default=BASE_DIR / "data" / "train_data.csv")
    parser.add_argument("--train-start-year", type=int, default=2022)
    parser.add_argument("--val-year", type=int, default=2024)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--seeds", type=int, default=3, help="신경망 후보 시드 수")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=BASE_DIR / "compare_runs" / "checkpoints",
        help="모델별 체크포인트를 모을 공통 실험 폴더",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=BASE_DIR / "compare_runs" / "logs",
        help="모델별 실행 로그를 누적할 공통 폴더",
    )
    args = parser.parse_args()
    if min(args.window, args.seeds, args.epochs, args.patience) < 1:
        parser.error("--window, --seeds, --epochs, --patience는 1 이상이어야 합니다.")
    if args.train_start_year >= args.val_year:
        parser.error("--train-start-year는 --val-year보다 앞서야 합니다.")

    args.data = args.data.resolve()
    if not args.data.is_file():
        parser.error(
            f"학습 데이터가 없습니다: {args.data}\n"
            "먼저 2022년 안양대 격자 파일을 data/grid에 준비한 뒤 "
            "`python prepare_training_data.py --grid-dir data/grid`를 실행하세요."
        )
    experiment_dir = args.experiment_dir.resolve()
    model_dir = experiment_dir / args.model
    model_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now().astimezone()
    run_id = started.strftime("%Y%m%d_%H%M%S_%f")
    model_log_dir = args.log_dir.resolve() / args.model
    model_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_log_dir / f"train_{run_id}.log"
    metadata = {
        "status": "running",
        "run_id": run_id,
        "started_at": started.isoformat(timespec="seconds"),
        "model": args.model,
        "log_file": str(log_path),
        "settings": {
            "data": str(args.data),
            "train_val_sha256": train_val_sha256(
                args.data, args.train_start_year, args.val_year
            ),
            "train_start_year": args.train_start_year,
            "val_year": args.val_year,
            "window": args.window,
            "seeds": args.seeds,
            "epochs": args.epochs,
            "patience": args.patience,
            "device_requested": args.device,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions(["numpy", "pandas", "scikit-learn", "lightgbm", "torch"]),
        },
        "overall_score": {
            "version": OVERALL_SCORE_VERSION,
            "formula": OVERALL_SCORE_FORMULA,
            "mae_score": "100 × clip(1 - 체감 오차(°C) / 3, 0, 1)",
        },
    }
    write_json(model_dir / "metadata.json", metadata)

    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
            with contextlib.redirect_stdout(Tee(sys.stdout, log_file)), contextlib.redirect_stderr(Tee(sys.stderr, log_file)):
                print("=" * 64)
                print(f"단일 모델 학습: {TRAIN_MODEL_NAMES[args.model]}")
                print(f"체크포인트 폴더: {model_dir}")
                print(f"로그 파일: {log_path}")
                print("=" * 64)
                details, checkpoint_name = train_one(args, model_dir)
        metadata.update(details)
        metadata.update({
            "status": "completed",
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "files": [checkpoint_name, "metadata.json", "validation_metrics.csv",
                      "validation_season_metrics.csv", "validation_predictions.csv"],
        })
    except Exception as exc:
        metadata.update({
            "status": "failed",
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "error": f"{type(exc).__name__}: {exc}",
        })
        with log_path.open("a", encoding="utf-8") as log_file:
            traceback.print_exc(file=log_file)
        raise
    finally:
        write_json(model_dir / "metadata.json", metadata)


if __name__ == "__main__":
    main()
