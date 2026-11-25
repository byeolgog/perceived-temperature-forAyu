"""서버 없이 동작하는 웹(GitHub Pages) 버전용 모델·설정 내보내기.

검증으로 선택해 배포한 캠퍼스 모델과 작은 옷차림 분류 RF를 docs/model.js로 저장합니다.
배포 모델이 없을 때만 기존 축소 RF를 캠퍼스 보정 모델로 사용합니다.
브라우저가 이 파일을 읽어 서버 없이도 같은 방식으로 계산합니다. 설정값(옷 목록 등)도 함께 넣어
파이썬과 웹이 항상 같은 값을 쓰게 합니다.

사용법: python web_export.py   (config.py·outfit.py·train_data.csv 수정 후 다시 실행)
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import campus_model
import config
import model
import outfit
import shop
from train_campus import load_data

OUT = config.BASE_DIR / "docs" / "model.js"


def export_tree(est):
    """트리 하나 → 배열 (잎 노드의 확률 순서는 rf.classes_ 와 같음)."""
    t = est.tree_
    leaves = {}
    for i in range(t.node_count):
        if t.children_left[i] == -1:
            v = t.value[i][0]
            leaves[i] = [round(float(x), 3) for x in v / v.sum()]
    return {
        "f": [int(x) if x >= 0 else -1 for x in t.feature],
        "t": [round(float(x), 4) for x in t.threshold],
        "l": [int(x) for x in t.children_left],
        "r": [int(x) for x in t.children_right],
        "v": {str(k): v for k, v in leaves.items()},
    }


def export_reg_tree(est):
    """회귀 트리 하나 → 배열 (잎 노드 값 = 보정량)."""
    t = est.tree_
    return {
        "f": [int(x) if x >= 0 else -1 for x in t.feature],
        "t": [float(x) for x in t.threshold],
        "l": [int(x) for x in t.children_left],
        "r": [int(x) for x in t.children_right],
        "v": {str(i): float(t.value[i][0][0]) for i in range(t.node_count) if t.children_left[i] == -1},
    }


def export_rf_model(cm):
    return {
        "kind": "rf",
        "features": campus_model.FEATURES,
        "trees": {
            target: [export_reg_tree(tree) for tree in cm.models[target].estimators_]
            for target in campus_model.TARGETS
        },
    }


def export_lgb_node(node):
    if "leaf_value" in node:
        return {"v": float(node["leaf_value"])}
    return {
        "f": int(node["split_feature"]),
        "t": float(node["threshold"]),
        "d": bool(node.get("default_left", True)),
        "l": export_lgb_node(node["left_child"]),
        "r": export_lgb_node(node["right_child"]),
    }


def load_torch_checkpoint(path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def export_neural_model(path, key):
    """PyTorch 체크포인트를 브라우저에서 계산 가능한 순수 JSON 가중치로 변환한다."""
    saved = load_torch_checkpoint(path)
    state = {
        name: tensor.detach().cpu().numpy().tolist()
        for name, tensor in saved["state_dict"].items()
    }
    return {
        "kind": key,
        "features": list(saved["sequence_features"]),
        "window": int(saved["window"]),
        "input_mean": np.asarray(saved["input_mean"]).tolist(),
        "input_std": np.asarray(saved["input_std"]).tolist(),
        "target_mean": np.asarray(saved["target_mean"]).tolist(),
        "target_std": np.asarray(saved["target_std"]).tolist(),
        "layers": 2,
        "blocks": 3,
        "state": state,
    }


def export_selected_campus(manifest_path: Path = config.DEPLOYED_CAMPUS_MANIFEST):
    """배포 manifest의 선택 모델을 정적 사이트용 순수 JSON 트리로 변환한다."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    key = manifest["selected_model"]
    path = root / manifest["checkpoint"]
    common = {
        "selected_model": key,
        "name": manifest["selected_model_name"],
        "validation_year": manifest["validation_year"],
        "validation_overall_score": manifest["validation_overall_score"],
        "score_version": manifest["score_version"],
        "history_hours": manifest.get("history_hours", 1),
    }
    if key == "rf":
        selected = export_rf_model(campus_model.CampusModel.load(path))
    elif key == "rf-lag":
        saved = joblib.load(path)
        selected = {
            "kind": "rf-lag",
            "features": saved["features"],
            "trees": {
                target: [export_reg_tree(tree) for tree in saved["models"][index].estimators_]
                for index, target in enumerate(campus_model.TARGETS)
            },
        }
    elif key == "lightgbm":
        saved = joblib.load(path)
        selected = {
            "kind": "lightgbm",
            "features": saved["features"],
            "trees": {
                target: [
                    export_lgb_node(info["tree_structure"])
                    for info in saved["models"][index].booster_.dump_model()["tree_info"]
                ]
                for index, target in enumerate(campus_model.TARGETS)
            },
        }
    elif key in {"lstm", "tsmixer"}:
        selected = export_neural_model(path, key)
    else:
        raise RuntimeError(f"알 수 없는 선택 모델입니다: {key}")

    fallback_name = manifest.get("fallback_checkpoint")
    if key != "rf" and not fallback_name:
        raise RuntimeError("시계열 모델의 정적 배포에는 RF fallback 체크포인트가 필요합니다.")
    fallback = (
        export_rf_model(campus_model.CampusModel.load(root / fallback_name))
        if fallback_name else selected
    )
    return {**common, **selected, "fallback": fallback}


def export_campus(manifest_path: Path = config.DEPLOYED_CAMPUS_MANIFEST):
    """배포 모델이 없을 때만 기존 축소 RF를 만든다."""
    selected = export_selected_campus(manifest_path)
    if selected is not None:
        return selected
    cm = campus_model.CampusModel(n_estimators=20, max_leaf_nodes=64, max_depth=None, min_samples_leaf=20)
    cm.fit(load_data(config.CAMPUS_TRAIN_PATH))
    return {"name": "랜덤포레스트 축소판", **export_rf_model(cm), "fallback": export_rf_model(cm)}


def main():
    parser = argparse.ArgumentParser(description="정적 사이트 모델을 docs/model.js로 내보냅니다.")
    parser.add_argument("--manifest", type=Path, default=config.DEPLOYED_CAMPUS_MANIFEST)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    X, y, w = model.build_dataset(model.synthetic_weather())
    fb = model.load_feedback()
    if fb is not None:
        X, y, w = pd.concat([X, fb[0]]), np.concatenate([y, fb[1]]), np.concatenate([w, fb[2]])

    rf = RandomForestClassifier(n_estimators=30, max_leaf_nodes=64, min_samples_leaf=5, random_state=0)
    rf.fit(X, y, sample_weight=w)
    acc = (rf.predict(X) == y).mean()
    classes = [int(c) for c in rf.classes_]

    payload = {
        "features": model.FEATURES,
        "classes": classes,
        "trees": [export_tree(e) for e in rf.estimators_],
        "campus_model": export_campus(args.manifest),
        "config": {
            "campus": config.CAMPUS_NAME,
            "app_ref": [config.APP_REF_LAT, config.APP_REF_LON],
            "app_ref_name": config.APP_REF_NAME,
            "sensitivity": config.SENSITIVITY_OFFSET,
            "schedules": config.SCHEDULE_PRESETS,
            "levels": [{"level": l.level, "name": l.name, "categories": l.categories} for l in outfit.LEVELS],
            "category_emoji": outfit.CATEGORY_EMOJI,
            "item_emoji": outfit.ITEM_EMOJI,
            "rain_extra": outfit.RAIN_EXTRA,
            "search_links": shop.SEARCH_LINKS,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    js = "// python web_export.py 로 생성된 파일 — 직접 수정하지 마세요\nwindow.ONULCHUPNA = "
    args.out.write_text(js + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    campus = payload["campus_model"]
    print(f"웹 모델 저장: {args.out} ({args.out.stat().st_size / 1024:.0f}KB, 학습 정확도 {acc:.1%})")
    print(f"정적 캠퍼스 모델: {campus.get('name')} ({campus.get('kind')})")


if __name__ == "__main__":
    main()
