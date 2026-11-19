"""서버 없이 동작하는 웹(GitHub Pages) 버전용 모델·설정 내보내기.

작은 Random Forest 두 종류를 학습해 docs/model.js 로 저장합니다.
  - 캠퍼스 보정 모델 (만안 → 안양대 기온·습도·풍속 보정량, 회귀) 축소판
  - 옷차림 분류 모델 축소판
브라우저가 이 파일을 읽어 서버 없이도 같은 방식으로 계산합니다. 설정값(옷 목록 등)도 함께 넣어
파이썬과 웹이 항상 같은 값을 쓰게 합니다.

사용법: python web_export.py   (config.py·outfit.py·train_data.csv 수정 후 다시 실행)
"""
import json

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
        "t": [round(float(x), 4) for x in t.threshold],
        "l": [int(x) for x in t.children_left],
        "r": [int(x) for x in t.children_right],
        "v": {str(i): round(float(t.value[i][0][0]), 3) for i in range(t.node_count) if t.children_left[i] == -1},
    }


def export_campus():
    """캠퍼스 보정 모델 축소판 (나무 20그루 × 정답 3개)."""
    cm = campus_model.CampusModel(n_estimators=20, max_leaf_nodes=64, max_depth=None, min_samples_leaf=20)
    cm.fit(load_data(config.CAMPUS_TRAIN_PATH))
    return {
        "features": campus_model.FEATURES,
        "targets": campus_model.TARGETS,
        "trees": {t: [export_reg_tree(e) for e in cm.models[t].estimators_] for t in campus_model.TARGETS},
    }


def main():
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
        "campus_model": export_campus(),
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
    OUT.parent.mkdir(exist_ok=True)
    js = "// python web_export.py 로 생성된 파일 — 직접 수정하지 마세요\nwindow.ONULCHUPNA = "
    OUT.write_text(js + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"웹 모델 저장: {OUT} ({OUT.stat().st_size / 1024:.0f}KB, 학습 정확도 {acc:.1%})")


if __name__ == "__main__":
    main()
