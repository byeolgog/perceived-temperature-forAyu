"""정적 model.js 예측이 배포된 Python 체크포인트와 일치하는지 확인한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from campus_model import BASES, FEATURES, TARGETS, tidy
from compare_models import BASE_DIR, lag_features, load_hourly
from deployed_model import DeployedCampusModel
from static_neural import neural_delta


def load_payload(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return json.loads(text[text.index("{"): text.rindex("}") + 1])["campus_model"]


def rf_tree(tree: dict, values: np.ndarray) -> float:
    node = 0
    while tree["f"][node] != -1:
        node = tree["l"][node] if values[tree["f"][node]] <= tree["t"][node] else tree["r"][node]
    leaves = tree["v"]
    return float(leaves[str(node)] if isinstance(leaves, dict) else leaves[node])


def lgb_tree(node: dict, values: np.ndarray) -> float:
    while "v" not in node:
        value = values[node["f"]]
        node = (node["l"] if node["d"] else node["r"]) if not np.isfinite(value) \
            else (node["l"] if value <= node["t"] else node["r"])
    return float(node["v"])


def delta(model: dict, values: np.ndarray, target: str) -> float:
    trees = model["trees"][target]
    if model.get("kind") == "lightgbm":
        return sum(lgb_tree(tree, values) for tree in trees)
    return float(np.mean([rf_tree(tree, values) for tree in trees]))


def static_predict(payload: dict, frame) -> np.ndarray:
    base = frame[BASES].to_numpy(float)
    fallback = payload.get("fallback", payload)
    current = frame[FEATURES].to_numpy(float)
    prediction = tidy(base + np.asarray([
        [delta(fallback, row, target) for target in TARGETS] for row in current
    ]))
    if payload.get("kind", "rf") == "rf":
        return tidy(base + np.asarray([
            [delta(payload, row, target) for target in TARGETS] for row in current
        ]))
    if payload["kind"] in {"rf-lag", "lightgbm"}:
        lag = lag_features(frame)
        features = payload["features"]
        valid = lag[features].notna().all(axis=1).to_numpy()
        for row_index in np.where(valid)[0]:
            values = lag.iloc[row_index][features].to_numpy(float)
            prediction[row_index] = tidy(np.asarray([base[row_index] + [
                delta(payload, values, target) for target in TARGETS
            ]]))[0]
    elif payload["kind"] in {"lstm", "tsmixer"}:
        window = int(payload["window"])
        raw = frame[payload["features"]].to_numpy(float)
        for row_index in range(window - 1, len(frame)):
            sequence = raw[row_index - window + 1:row_index + 1]
            if np.isfinite(sequence).all():
                prediction[row_index] = tidy(np.asarray([
                    base[row_index] + neural_delta(payload, sequence)
                ]))[0]
    else:
        raise RuntimeError(f"정적 검증 미지원 모델: {payload['kind']}")
    return prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-js", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=BASE_DIR / "data" / "prepared" / "model_data.csv")
    parser.add_argument("--tolerance", type=float, default=0.15)
    args = parser.parse_args()

    frame = load_hourly(args.data.resolve()).tail(72).copy()
    payload = load_payload(args.model_js.resolve())
    python_prediction = DeployedCampusModel(args.manifest.resolve()).predict(frame)
    browser_prediction = static_predict(payload, frame)
    difference = np.abs(python_prediction - browser_prediction)
    max_difference = float(difference.max())
    if max_difference > args.tolerance:
        raise RuntimeError(f"정적 모델 오차가 허용값을 넘었습니다: {max_difference:.6f}")
    print(f"정적 모델 일치 확인: {payload.get('name')} · 최대 차이 {max_difference:.6f}")


if __name__ == "__main__":
    main()
