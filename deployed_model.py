"""검증으로 선택해 배포한 캠퍼스 모델의 앱 추론 어댑터."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config
from campus_model import BASES, CampusModel, add_features, tidy
from compare_models import SEQ_FEATURES, build_nets, lag_features


def _load_torch(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class DeployedCampusModel:
    """모델 종류와 무관하게 ``predict(DataFrame)``를 제공한다."""

    def __init__(self, manifest_path: Path | None = None):
        self.manifest_path = Path(manifest_path or config.DEPLOYED_CAMPUS_MANIFEST)
        self.root = self.manifest_path.parent
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.key = self.manifest["selected_model"]
        self.name = self.manifest["selected_model_name"]
        self.checkpoint_path = self.root / self.manifest["checkpoint"]
        self.checkpoint = None
        self.fallback = None
        self.net = None
        self.last_selected_rows = 0
        self.last_fallback_rows = 0
        self._load()

    def _load(self):
        if self.key == "rf":
            self.checkpoint = CampusModel.load(self.checkpoint_path)
        elif self.key in {"rf-lag", "lightgbm"}:
            self.checkpoint = joblib.load(self.checkpoint_path)
        else:
            self.checkpoint = _load_torch(self.checkpoint_path)
            import torch

            LSTMNet, TSMixer = build_nets()
            channels = len(SEQ_FEATURES)
            self.net = (
                LSTMNet(channels)
                if self.key == "lstm"
                else TSMixer(self.checkpoint["window"], channels)
            )
            self.net.load_state_dict(self.checkpoint["state_dict"])
            self.net.eval()
            self.net.to("cpu")

        fallback_name = self.manifest.get("fallback_checkpoint")
        if fallback_name:
            self.fallback = CampusModel.load(self.root / fallback_name)

    @property
    def history_hours(self) -> int:
        return int(self.manifest.get("history_hours", 1))

    def _hourly(self, frame: pd.DataFrame):
        source = frame.copy()
        if "tm" not in source and "time" in source:
            source["tm"] = source["time"]
        if "tm" not in source:
            raise ValueError("배포 모델 입력에는 tm 또는 time 열이 필요합니다.")
        source["tm"] = pd.to_datetime(source["tm"])
        source["_original_order"] = np.arange(len(source))
        original = source[["tm", "_original_order"]].copy()
        source = source.sort_values("tm").drop_duplicates("tm", keep="last").set_index("tm")
        hourly_index = pd.date_range(source.index.min(), source.index.max(), freq="h")
        hourly = source.reindex(hourly_index)
        hourly[BASES] = hourly[BASES].interpolate(limit_direction="both")
        hourly["month"] = hourly.index.month
        hourly["hour"] = hourly.index.hour
        hourly = add_features(hourly.rename_axis("tm").reset_index())
        positions = pd.Series(np.arange(len(hourly)), index=hourly.tm)
        original_positions = original.tm.map(positions).to_numpy(int)
        restore = np.argsort(original._original_order.to_numpy())
        return hourly, original_positions, restore

    def _fallback_prediction(self, hourly: pd.DataFrame) -> np.ndarray:
        if self.fallback is not None:
            return self.fallback.predict(hourly)
        return hourly[BASES].to_numpy(float)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        hourly, original_positions, restore = self._hourly(frame)
        base = hourly[BASES].to_numpy(float)
        prediction = self._fallback_prediction(hourly)
        selected_mask = np.zeros(len(hourly), dtype=bool)

        if self.key == "rf":
            prediction = self.checkpoint.predict(hourly)
            selected_mask[:] = True
        elif self.key in {"rf-lag", "lightgbm"}:
            features = self.checkpoint["features"]
            lag = lag_features(hourly)
            selected_mask = lag[features].notna().all(axis=1).to_numpy()
            if selected_mask.any():
                delta = np.column_stack([
                    model.predict(lag.loc[selected_mask, features])
                    for model in self.checkpoint["models"]
                ])
                prediction[selected_mask] = tidy(base[selected_mask] + delta)
        else:
            import torch

            saved = self.checkpoint
            window = int(saved["window"])
            raw = hourly[SEQ_FEATURES].to_numpy(np.float32)
            normalized = (
                (raw - np.asarray(saved["input_mean"], dtype=np.float32))
                / np.asarray(saved["input_std"], dtype=np.float32)
            )
            rows = np.arange(window - 1, len(hourly))
            if len(rows):
                index = rows[:, None] + np.arange(-window + 1, 1)[None, :]
                sequences = normalized[index]
                finite = np.isfinite(sequences).all(axis=(1, 2))
                rows, sequences = rows[finite], sequences[finite]
                if len(rows):
                    with torch.no_grad():
                        scaled = self.net(torch.from_numpy(sequences)).cpu().numpy()
                    delta = (
                        scaled * np.asarray(saved["target_std"])
                        + np.asarray(saved["target_mean"])
                    )
                    prediction[rows] = tidy(base[rows] + delta)
                    selected_mask[rows] = True

        self.last_selected_rows = int(selected_mask[original_positions].sum())
        self.last_fallback_rows = int(len(original_positions) - self.last_selected_rows)
        return tidy(prediction[original_positions][restore])


def deployment_status(manifest_path: Path | None = None) -> dict:
    path = Path(manifest_path or config.DEPLOYED_CAMPUS_MANIFEST)
    if not path.is_file():
        return {"deployed": False, "mode": "legacy_rf"}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {
        "deployed": True,
        "mode": "selected_checkpoint",
        "selected_model": manifest["selected_model"],
        "selected_model_name": manifest["selected_model_name"],
        "validation_year": manifest["validation_year"],
        "validation_overall_score": manifest["validation_overall_score"],
        "score_version": manifest["score_version"],
        "history_hours": manifest.get("history_hours", 1),
        "has_rf_fallback": bool(manifest.get("fallback_checkpoint")),
    }
