"""2024년 검증 1위 체크포인트를 Flask 앱용 모델로 배포하고 추론을 확인한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

import config
from compare_models import BASE_DIR, OVERALL_SCORE_VERSION, TRAIN_MODEL_NAMES, load_hourly
from deployed_model import DeployedCampusModel


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_checkpoint(source: Path, output_dir: Path, stem: str) -> Path:
    destination = output_dir / f"{stem}{source.suffix}"
    shutil.copy2(source, destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description="검증 1위 모델을 앱에 배포합니다.")
    parser.add_argument(
        "--experiment-dir", type=Path,
        default=BASE_DIR / "compare_runs" / "checkpoints",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=config.DEPLOYED_CAMPUS_MODEL_DIR,
    )
    parser.add_argument(
        "--data", type=Path, default=BASE_DIR / "data" / "prepared" / "model_data.csv",
    )
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    selection_path = experiment_dir / "selection.json"
    if not selection_path.is_file():
        raise SystemExit("selection.json이 없습니다. 먼저 compare_checkpoints.py --select-only을 실행하세요.")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("score_version") != OVERALL_SCORE_VERSION:
        raise SystemExit(
            f"점수 버전 불일치: {selection.get('score_version')} != {OVERALL_SCORE_VERSION}. "
            "새 공식으로 모든 모델을 다시 학습·선택하세요."
        )

    selected_key = selection["selected_model"]
    source = experiment_dir / selection["checkpoint"]
    if not source.is_file():
        raise SystemExit(f"선택 체크포인트가 없습니다: {source}")
    if sha256(source) != selection["checkpoint_sha256"]:
        raise SystemExit("선택 체크포인트 해시가 selection.json과 다릅니다.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    deployed = copy_checkpoint(source, output_dir, f"selected_{selected_key}")

    fallback = None
    if selected_key != "rf":
        fallback_source = experiment_dir / "rf" / "best.joblib"
        if not fallback_source.is_file():
            raise SystemExit(
                "시계열 최고 모델의 첫 구간을 보완할 RF 체크포인트가 없습니다. "
                "먼저 python train_model.py --model rf 를 실행하세요."
            )
        fallback = copy_checkpoint(fallback_source, output_dir, "fallback_rf")

    history_hours = 1 if selected_key == "rf" else 24
    if selected_key in {"lstm", "tsmixer"}:
        from deployed_model import _load_torch
        history_hours = int(_load_torch(deployed)["window"])

    manifest = {
        "schema_version": 1,
        "deployed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selected_model": selected_key,
        "selected_model_name": selection.get("selected_model_name", TRAIN_MODEL_NAMES[selected_key]),
        "checkpoint": deployed.name,
        "checkpoint_sha256": sha256(deployed),
        "fallback_checkpoint": fallback.name if fallback else None,
        "fallback_checkpoint_sha256": sha256(fallback) if fallback else None,
        "validation_year": selection["validation_year"],
        "validation_overall_score": selection["validation_overall_score"],
        "score_version": selection["score_version"],
        "history_hours": history_hours,
        "selection_source": str(selection_path),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    data_path = args.data.resolve()
    if not data_path.is_file():
        raise SystemExit(f"앱 추론 확인용 데이터가 없습니다: {data_path}")
    hourly = load_hourly(data_path)
    usable = hourly[hourly["observed"] & hourly["aws_ta"].notna()].tail(72)
    smoke = usable[["tm", "aws_ta", "aws_hm", "aws_ws"]].copy()
    app_model = DeployedCampusModel(manifest_path)
    prediction = app_model.predict(smoke)
    if prediction.shape != (len(smoke), 3) or not np.isfinite(prediction).all():
        raise RuntimeError("배포 모델의 앱 추론 결과가 올바르지 않습니다.")

    print(f"앱 배포 모델: {app_model.name}")
    print(f"검증 종합점수: {manifest['validation_overall_score']:.2f} ({manifest['score_version']})")
    print(f"배포 폴더: {output_dir}")
    print(
        f"추론 확인: {len(smoke)}시간 중 선택 모델 {app_model.last_selected_rows}시간 / "
        f"RF fallback {app_model.last_fallback_rows}시간"
    )
    print("Flask 앱 연동 확인: python server.py 실행 후 /api/model-status 확인")


if __name__ == "__main__":
    main()
