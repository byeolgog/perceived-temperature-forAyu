"""캠퍼스 보정 모델 비교: 랜덤포레스트 vs LightGBM vs LSTM vs TSMixer.

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install lightgbm
    python compare_models.py --train all                  # 다섯 모델 학습 → 최종 비교
    python compare_models.py --train rf rf-lag lightgbm   # 나무 모델만 학습 → 비교
    python compare_models.py --train lstm tsmixer --seeds 3

모두 train_campus.py 와 같은 방식으로 채점한다 (정답 = 안양대 500m 격자, 체감은 기상청 공식).
모든 모델은 '안양대 − 만안' 보정량을 맞히고, 차이는 입력에 있다:

- RF            : 지금 시각 만안 값 1개 (지금 운영 중인 모델과 같음)
- RF + 지연특성 : 지금 + 직전 몇 시간 값·변화량·이동평균  (특성 효과만 따로 보려고)
- LightGBM      : RF + 지연특성과 같은 입력, 그래디언트 부스팅 (가볍고 빠름)
- LSTM          : 직전 window 시간의 만안 시퀀스 그대로
- TSMixer       : 같은 시퀀스, MLP로 시간축·변수축을 번갈아 섞는 2023년 경량 시계열 모델

운영 중 입력은 만안 '예보' 시계열이라 직전 시간 값도 예보/실황으로 채울 수 있다.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import platform
import sys
import time
import traceback
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from campus_model import BASES, TARGETS, CampusModel, add_features, kma_feels, tidy

BASE_DIR = Path(__file__).resolve().parent
SEQ_FEATURES = ["aws_ta", "aws_hm", "aws_ws", "hour_sin", "hour_cos", "month_sin", "month_cos"]
REF_MODEL = "RF (지금 운영)"
OVERALL_SCORE_VERSION = "overall_v2_season_balanced"
OVERALL_SCORE_FORMULA = {
    "±1°C 적중(%)": 0.40,
    "봄(3~5월) ±1°C(%)": 0.075,
    "여름(6~8월) ±1°C(%)": 0.075,
    "가을(9~11월) ±1°C(%)": 0.075,
    "겨울(12~2월) ±1°C(%)": 0.075,
    "±2°C 적중(%)": 0.15,
    "체감 MAE 점수": 0.15,
}
MODEL_KEYS = {
    "보정 안 함(날씨앱)": "uncorrected",
    REF_MODEL: "rf_current",
    "RF + 지연특성": "rf_lag",
    "LightGBM + 지연특성": "lightgbm_lag",
    "LSTM": "lstm",
    "TSMixer": "tsmixer",
}
TRAIN_MODEL_ORDER = ("rf", "rf-lag", "lightgbm", "lstm", "tsmixer")
TRAIN_MODEL_NAMES = {
    "rf": REF_MODEL,
    "rf-lag": "RF + 지연특성",
    "lightgbm": "LightGBM + 지연특성",
    "lstm": "LSTM",
    "tsmixer": "TSMixer",
}

# ---------------- 데이터 ----------------

def load_hourly(path: Path) -> pd.DataFrame:
    """1시간 간격으로 다시 맞추고, 만안 입력의 짧은 빈칸(≤9시간)은 선형보간.
    observed=True 인 줄(원래 있던 줄)만 학습·채점에 쓴다."""
    df = pd.read_csv(path, parse_dates=["tm"], encoding="utf-8-sig").set_index("tm").sort_index()
    full = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="h"))
    full["observed"] = full["uni_ta"].notna()
    full[BASES] = full[BASES].interpolate(limit=9, limit_area="inside")
    full["month"], full["hour"] = full.index.month, full.index.hour
    full = add_features(full.rename_axis("tm").reset_index())
    return full


def lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[SEQ_FEATURES].copy()
    for c in BASES:
        s = df[c]
        for k in (1, 2, 3, 6, 12, 24):
            out[f"{c}_lag{k}"] = s.shift(k)
        out[f"{c}_d1"] = s - s.shift(1)
        out[f"{c}_d3"] = s - s.shift(3)
        for w in (3, 6, 24):
            out[f"{c}_mean{w}"] = s.rolling(w).mean()
        out[f"{c}_std6"] = s.rolling(6).std()
    out["ta_range24"] = df["aws_ta"].rolling(24).max() - df["aws_ta"].rolling(24).min()
    out["ws_max6"] = df["aws_ws"].rolling(6).max()
    return out


def windows(df: pd.DataFrame, rows: np.ndarray, L: int, mu, sd) -> np.ndarray:
    """rows 마다 [row-L+1 .. row] 구간의 시퀀스 (N, L, C)."""
    x = ((df[SEQ_FEATURES].to_numpy(np.float32) - mu) / sd).astype(np.float32)
    idx = rows[:, None] + np.arange(-L + 1, 1)[None, :]
    return x[idx]


# ---------------- 신경망 ----------------

def build_nets():
    import torch
    from torch import nn

    class LSTMNet(nn.Module):
        def __init__(self, c, hidden=64, layers=2, out=3, drop=0.1):
            super().__init__()
            self.lstm = nn.LSTM(c, hidden, layers, batch_first=True, dropout=drop)
            self.head = nn.Sequential(nn.Linear(hidden + c, 64), nn.GELU(), nn.Linear(64, out))

        def forward(self, x):
            h, _ = self.lstm(x)
            # 마지막 은닉 상태 + 지금 시각 입력을 그대로 함께 (지금 값이 제일 중요해서)
            return self.head(torch.cat([h[:, -1], x[:, -1]], dim=1))

    class MixerBlock(nn.Module):
        def __init__(self, L, c, ff=64, drop=0.1):
            super().__init__()
            self.norm_t, self.norm_f = nn.LayerNorm(c), nn.LayerNorm(c)
            self.time = nn.Sequential(nn.Linear(L, L), nn.GELU(), nn.Dropout(drop))
            self.feat = nn.Sequential(nn.Linear(c, ff), nn.GELU(), nn.Dropout(drop), nn.Linear(ff, c), nn.Dropout(drop))

        def forward(self, x):                      # x: (B, L, C)
            x = x + self.time(self.norm_t(x).transpose(1, 2)).transpose(1, 2)   # 시간축 섞기
            return x + self.feat(self.norm_f(x))                                # 변수축 섞기

    class TSMixer(nn.Module):
        def __init__(self, L, c, hidden=32, blocks=3, out=3, drop=0.1):
            super().__init__()
            self.proj = nn.Linear(c, hidden)
            self.blocks = nn.Sequential(*[MixerBlock(L, hidden, 2 * hidden, drop) for _ in range(blocks)])
            self.head = nn.Sequential(nn.Linear(L * hidden + c, 64), nn.GELU(), nn.Linear(64, out))

        def forward(self, x):
            z = self.blocks(self.proj(x)).flatten(1)
            return self.head(torch.cat([z, x[:, -1]], dim=1))

    return LSTMNet, TSMixer


def resolve_torch_device(requested: str = "auto") -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다. Colab 런타임을 GPU로 바꾸거나 --device cpu를 사용하세요.")
    return requested


def train_net(make, Xtr, Ytr, Xte, seed, epochs=60, patience=8, lr=2e-3, bs=256,
              device="auto", Xval=None, Yval=None, progress=None):
    import torch
    from torch import nn

    device = resolve_torch_device(device)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    if Xval is None or Yval is None:
        # 이전 호출 호환: 명시적 검증 세트가 없으면 학습 끝 10%를 사용한다.
        n_val = len(Xtr) // 10
        Xt, Yt = torch.from_numpy(Xtr[:-n_val]), torch.from_numpy(Ytr[:-n_val])
        Xv, Yv = torch.from_numpy(Xtr[-n_val:]), torch.from_numpy(Ytr[-n_val:])
    else:
        Xt, Yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
        Xv, Yv = torch.from_numpy(Xval), torch.from_numpy(Yval)
    net = make().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.HuberLoss(delta=1.0)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(len(Xt))
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            xb, yb = Xt[b].to(device), Yt[b].to(device)
            opt.zero_grad()
            loss_fn(net(xb), yb).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            v = loss_fn(net(Xv.to(device)), Yv.to(device)).item()
        should_stop = False
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            should_stop = bad >= patience
        if progress is not None:
            progress(ep + 1, epochs, v, best, bad, opt.param_groups[0]["lr"])
        if should_stop:
            break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred = net(torch.from_numpy(Xte).to(device)).cpu().numpy()
    n_params = sum(p.numel() for p in net.parameters())
    return pred, n_params, ep + 1, best, best_state


# ---------------- 출력 ----------------

def _w(text: str) -> int:
    """터미널에서 차지하는 칸 수 (한글은 2칸)."""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int, right: bool = False) -> str:
    gap = " " * max(0, width - _w(text))
    return gap + text if right else text + gap


# (열 이름, score 키, 형식, 클수록 좋은가)
COLUMNS = [
    ("종합 점수", "종합 점수", "{:.1f}", True),
    ("체감 ±1°C", "±1°C 적중(%)", "{:.1f}%", True),
    ("사계절 평균", "사계절 평균 ±1°C(%)", "{:.1f}%", True),
    ("체감 ±2°C", "±2°C 적중(%)", "{:.1f}%", True),
    ("겨울공식 ±1°C", "겨울공식 기간(10~4월) ±1°C(%)", "{:.1f}%", True),
    ("체감 오차", "체감 오차(°C)", "{:.2f}°C", False),
    ("기온 오차", "기온 오차(°C)", "{:.2f}°C", False),
    ("습도 오차", "습도 오차(%)", "{:.2f}%", False),
    ("풍속 오차", "풍속 오차(m/s)", "{:.2f}", False),
]


def print_report(results: dict, info: dict, ref: str, title: str) -> None:
    names = list(results)
    models = [n for n in names if n in info]          # 기준선(날씨앱)은 1등 후보에서 뺌
    best = {key: (max if hi else min)(models, key=lambda n: results[n][key]) for _, key, _, hi in COLUMNS}
    has_ref = ref in results

    name_w = max(_w(n) for n in names) + 2
    col_widths = [max(11, _w(label) + 2) for label, *_ in COLUMNS]
    header = (_pad("모델", name_w) + "".join(
                  _pad(column[0], width, right=True)
                  for column, width in zip(COLUMNS, col_widths)
              )
              + _pad("RF 대비", 10, right=True) + "   학습")
    line = "─" * _w(header)
    print(f"\n{title}\n{line}\n{header}\n{line}")
    for n in names:
        r = results[n]
        cells = []
        for (_, key, fmt, _), width in zip(COLUMNS, col_widths):
            v = fmt.format(r[key]) + ("★" if best[key] == n else " ")
            cells.append(_pad(v, width, right=True))
        diff = r["±1°C 적중(%)"] - results[ref]["±1°C 적중(%)"] if has_ref else np.nan
        diff_s = ("기준" if n == ref else f"{diff:+.1f}%p") if has_ref else "-"
        print(_pad(n, name_w) + "".join(cells) + _pad(diff_s, 10, right=True) + "   " + info.get(n, "-"))
    print(line)
    top = max(models, key=lambda n: results[n]["종합 점수"])
    lead = (f", 지금 운영 RF보다 "
            f"{results[top]['±1°C 적중(%)'] - results[ref]['±1°C 적중(%)']:+.1f}%p") if has_ref else ""
    print(f"★ = 열별 1등 · 'RF 대비' = 체감 ±1°C 적중률 차이\n"
          f"→ 종합 1등: {top}  ({results[top]['종합 점수']:.1f}점 · "
          f"±1°C {results[top]['±1°C 적중(%)']:.1f}%{lead})")
    print("  종합 점수 = 연중 ±1°C 40% + 사계절 균형 ±1°C 30% + ±2°C 15% + 체감 MAE 점수 15%")


class Tee:
    """터미널 출력과 UTF-8 로그 파일에 같은 내용을 쓴다."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False


class Step:
    """모델별 시작과 완료를 로그에서 찾기 쉬운 두 줄로 표시한다."""

    def __init__(self, total: int):
        self.total, self.i = total, 0

    def __call__(self, name: str):
        self.i += 1
        self.name, self.t0 = name, time.time()
        print(f"  [{self.i}/{self.total}] {name}", flush=True)
        return self

    def done(self, extra: str = "") -> str:
        sec = time.time() - self.t0
        self.last_seconds = sec
        took = f"{sec:.0f}초" + (f" · {extra}" if extra else "")
        print(f"        완료 | {took}", flush=True)
        return took


# ---------------- 채점 ----------------

def score(test: pd.DataFrame, p: np.ndarray) -> dict:
    real = kma_feels(test.month, test.uni_ta, test.uni_hm, test.uni_ws)
    err = np.abs(kma_feels(test.month, p[:, 0], p[:, 1], p[:, 2]) - real)
    month = test.month.to_numpy()
    winter_formula_period = ~test.month.between(5, 9).to_numpy()
    season_masks = {
        "봄(3~5월) ±1°C(%)": np.isin(month, [3, 4, 5]),
        "여름(6~8월) ±1°C(%)": np.isin(month, [6, 7, 8]),
        "가을(9~11월) ±1°C(%)": np.isin(month, [9, 10, 11]),
        "겨울(12~2월) ±1°C(%)": np.isin(month, [12, 1, 2]),
    }
    hit1 = (err <= 1).mean() * 100
    hit2 = (err <= 2).mean() * 100
    season_hit1 = {
        name: ((err[mask] <= 1).mean() * 100 if mask.any() else np.nan)
        for name, mask in season_masks.items()
    }
    season_balanced_hit1 = float(np.nanmean(list(season_hit1.values())))
    winter_formula_hit1 = (err[winter_formula_period] <= 1).mean() * 100
    feel_mae = err.mean()
    mae_score = np.clip(1 - feel_mae / 3, 0, 1) * 100
    overall = (
        0.40 * hit1
        + 0.30 * season_balanced_hit1
        + 0.15 * hit2
        + 0.15 * mae_score
    )
    return {
        "종합 점수": overall,
        "기온 오차(°C)": np.abs(p[:, 0] - test.uni_ta).mean(),
        "습도 오차(%)": np.abs(p[:, 1] - test.uni_hm).mean(),
        "풍속 오차(m/s)": np.abs(p[:, 2] - test.uni_ws).mean(),
        "체감 오차(°C)": feel_mae,
        "±1°C 적중(%)": hit1,
        "사계절 평균 ±1°C(%)": season_balanced_hit1,
        **season_hit1,
        "±2°C 적중(%)": hit2,
        "겨울공식 기간(10~4월) ±1°C(%)": winter_formula_hit1,
    }


def result_frame(results: dict, info: dict, timings: dict, ref: str = REF_MODEL) -> pd.DataFrame:
    """화면 표와 같은 값을 저장하기 좋은 표로 만든다."""
    frame = pd.DataFrame.from_dict(results, orient="index")
    frame = frame[[key for _, key, _, _ in COLUMNS]]
    frame.index.name = "model"
    frame["RF 대비(%p)"] = (
        frame["±1°C 적중(%)"] - frame.loc[ref, "±1°C 적중(%)"] if ref in frame.index else np.nan
    )
    frame["학습 시간(초)"] = pd.Series(timings)
    frame["학습 정보"] = pd.Series(info).reindex(frame.index).fillna("-")
    return frame


def season_frame(test: pd.DataFrame, predictions: dict) -> pd.DataFrame:
    """모델별 계절 ±1°C 적중률을 계산한다."""
    season = np.select(
        [test.month.isin([12, 1, 2]), test.month.isin([3, 4, 5]), test.month.isin([6, 7, 8])],
        ["겨울", "봄", "여름"],
        default="가을",
    )
    real = kma_feels(test.month, test.uni_ta, test.uni_hm, test.uni_ws)
    rows = {}
    for name, pred in predictions.items():
        estimated = kma_feels(test.month, pred[:, 0], pred[:, 1], pred[:, 2])
        hit = pd.DataFrame({"season": season, "hit": np.abs(estimated - real) <= 1})
        rows[name] = (hit.groupby("season").hit.mean() * 100).to_dict()
    frame = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=["봄", "여름", "가을", "겨울"])
    frame.index.name = "model"
    return frame


def prediction_frame(test: pd.DataFrame, predictions: dict) -> pd.DataFrame:
    """시험 기간 실제값과 모델별 예측값을 시간별로 펼친다."""
    out = test[["tm", *BASES, *TARGETS]].reset_index(drop=True).copy()
    out["real_feels"] = kma_feels(test.month, test.uni_ta, test.uni_hm, test.uni_ws)
    for name, pred in predictions.items():
        key = MODEL_KEYS[name]
        out[[f"{key}_ta", f"{key}_hm", f"{key}_ws"]] = pred
        out[f"{key}_feels"] = kma_feels(test.month, pred[:, 0], pred[:, 1], pred[:, 2])
    return out


def package_versions(names: list[str]) -> dict:
    found = {}
    for name in names:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = None
    return found


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def save_results(run_dir: Path, output_root: Path, run_id: str, args, summary: pd.DataFrame,
                 seasons: pd.DataFrame, predictions: pd.DataFrame) -> None:
    """실행별 상세 파일과 여러 실행을 모은 history.csv를 저장한다."""
    summary.to_csv(run_dir / "metrics.csv", encoding="utf-8-sig", float_format="%.4f")
    seasons.to_csv(run_dir / "season_metrics.csv", encoding="utf-8-sig", float_format="%.4f")
    predictions.to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig", float_format="%.4f")

    history = summary.reset_index()
    history.insert(0, "run_id", run_id)
    history.insert(1, "score_version", OVERALL_SCORE_VERSION)
    history.insert(2, "train_start_year", args.train_start_year)
    history.insert(3, "val_year", args.val_year)
    history.insert(4, "test_year", args.test_year)
    history.insert(5, "window", args.window)
    history.insert(6, "seeds", args.seeds)
    history_path = output_root / "history.csv"
    if history_path.exists():
        previous = pd.read_csv(history_path, encoding="utf-8")
        history = pd.concat([previous, history], ignore_index=True, sort=False)
    history.to_csv(history_path, index=False, encoding="utf-8", float_format="%.4f")


def run_comparison(args):
    """모델을 학습·채점하고 저장용 표들을 반환한다."""
    L = args.window
    selected = set(args.train_models)

    print("\n[1/3] 데이터 준비")
    df = load_hourly(args.data)
    lag = lag_features(df)
    year = df.tm.dt.year.to_numpy()
    # 시퀀스 앞부분이 모두 있는(보간 후 NaN 없는) 관측 줄만 사용 → 모든 모델이 같은 줄로 채점
    seq_ok = df[BASES].notna().all(axis=1).rolling(max(L, 25)).sum().eq(max(L, 25)).to_numpy()
    usable = df.observed.to_numpy() & seq_ok
    tr_rows = np.where(
        usable & (year >= args.train_start_year) & (year < args.val_year)
    )[0]
    va_rows = np.where(usable & (year == args.val_year))[0]
    te_rows = np.where(usable & (year == args.test_year))[0]
    train, val, test = df.iloc[tr_rows], df.iloc[va_rows], df.iloc[te_rows]
    base_te = test[BASES].to_numpy(float)
    D_tr = (train[TARGETS].to_numpy(float) - train[BASES].to_numpy(float))
    D_val = val[TARGETS].to_numpy(float) - val[BASES].to_numpy(float)
    years = sorted(df.loc[df.observed, "tm"].dt.year.unique().tolist())
    missing_train_years = sorted(
        set(range(args.train_start_year, args.val_year)) - set(years)
    )
    if missing_train_years or not len(train) or not len(val) or not len(test):
        raise ValueError(
            f"학습({args.train_start_year}~{args.val_year - 1}: {len(train):,}줄), "
            f"검증({args.val_year}: {len(val):,}줄), 시험({args.test_year}: {len(test):,}줄)이 "
            f"모두 필요합니다. 누락 학습 연도: {missing_train_years or '없음'}, 현재 연도: {years}"
        )
    print(f"      학습 {len(train):,}시간 ({args.train_start_year}~{args.val_year - 1}년)")
    print(f"      검증 {len(val):,}시간 ({args.val_year}년)")
    print(f"      시험 {len(test):,}시간 ({args.test_year}년)")
    print(f"      시퀀스 입력 직전 {L}시간")
    print(f"      선택 모델 {', '.join(TRAIN_MODEL_NAMES[key] for key in args.train_models)}")

    print("\n[2/3] 모델 학습")
    results, predictions, info, timings = {}, {}, {}, {}
    results["보정 안 함(날씨앱)"] = score(test, base_te)
    predictions["보정 안 함(날씨앱)"] = base_te
    step = Step(len(selected))

    if "rf" in selected:
        step(REF_MODEL)
        rf = CampusModel().fit(train)
        predictions[REF_MODEL] = rf.predict(test)
        results[REF_MODEL] = score(test, predictions[REF_MODEL])
        info[REF_MODEL] = step.done()
        timings[REF_MODEL] = step.last_seconds

    if "rf-lag" in selected:
        from sklearn.ensemble import RandomForestRegressor
        name = TRAIN_MODEL_NAMES["rf-lag"]
        step(name)
        p = np.column_stack([
            RandomForestRegressor(n_estimators=100, min_samples_leaf=20, max_depth=14, n_jobs=-1, random_state=0)
            .fit(lag.iloc[tr_rows], D_tr[:, j]).predict(lag.iloc[te_rows]) for j in range(3)])
        predictions[name] = tidy(base_te + p)
        results[name] = score(test, predictions[name])
        info[name] = step.done()
        timings[name] = step.last_seconds

    if "lightgbm" in selected:
        import lightgbm as lgb
        name = TRAIN_MODEL_NAMES["lightgbm"]
        step(name)
        p = []
        for j in range(3):
            m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                  objective="huber", alpha=1.0, verbose=-1, random_state=0, n_jobs=-1)
            p.append(m.fit(lag.iloc[tr_rows], D_tr[:, j]).predict(lag.iloc[te_rows]))
        predictions[name] = tidy(base_te + np.column_stack(p))
        results[name] = score(test, predictions[name])
        info[name] = step.done()
        timings[name] = step.last_seconds

    if selected & {"lstm", "tsmixer"}:
        device = resolve_torch_device(args.device)
        print(f"      신경망 장치 {device}")
        LSTMNet, TSMixer = build_nets()
        feats = df[SEQ_FEATURES].to_numpy(np.float32)
        mu, sd = np.nanmean(feats[tr_rows], 0), np.nanstd(feats[tr_rows], 0) + 1e-6
        Xtr, Xte = windows(df, tr_rows, L, mu, sd), windows(df, te_rows, L, mu, sd)
        Xval = windows(df, va_rows, L, mu, sd)
        ymu, ysd = D_tr.mean(0), D_tr.std(0) + 1e-6
        Ytr = ((D_tr - ymu) / ysd).astype(np.float32)
        Yval = ((D_val - ymu) / ysd).astype(np.float32)
        C = len(SEQ_FEATURES)
        neural_models = {
            "lstm": ("LSTM", lambda: LSTMNet(C)),
            "tsmixer": ("TSMixer", lambda: TSMixer(L, C)),
        }
        for key in ("lstm", "tsmixer"):
            if key not in selected:
                continue
            name, make = neural_models[key]
            step(f"{name} (시드 {args.seeds}개)")
            preds = []
            for s in range(args.seeds):
                seed_t0 = time.time()
                pr, n_params, ep, val_loss, _ = train_net(
                    make, Xtr, Ytr, Xte, seed=s, device=device, Xval=Xval, Yval=Yval
                )
                preds.append(pr)
                print(f"        시드 {s + 1}/{args.seeds} | {ep}에폭 · "
                      f"검증 {val_loss:.4f} · {time.time() - seed_t0:.0f}초")
            delta = np.mean(preds, 0) * ysd + ymu
            predictions[name] = tidy(base_te + delta)
            results[name] = score(test, predictions[name])
            info[name] = step.done(f"파라미터 {n_params / 1000:.0f}K")
            timings[name] = step.last_seconds

    print("\n[3/3] 결과 비교")
    ref = REF_MODEL if "rf" in selected else None
    print_report(results, info, ref=ref,
                 title=f"결과: {args.test_year}년 {len(test):,}시간 시험 (높을수록 좋음: 적중 / 낮을수록 좋음: 오차)")
    return (
        result_frame(results, info, timings, ref=ref),
        season_frame(test, predictions),
        prediction_frame(test, predictions),
        len(train),
        len(val),
        len(test),
    )


def main():
    ap = argparse.ArgumentParser(description="캠퍼스 보정 모델을 같은 시험 데이터로 비교하고 결과를 기록합니다.")
    ap.add_argument("--data", default=BASE_DIR / "data" / "train_data.csv", type=Path)
    ap.add_argument("--train-start-year", default=2022, type=int)
    ap.add_argument("--val-year", default=2024, type=int)
    ap.add_argument("--test-year", default=2025, type=int)
    ap.add_argument("--window", default=24, type=int, help="LSTM/TSMixer가 보는 직전 시간 수")
    ap.add_argument("--seeds", default=1, type=int, help="신경망을 시드 몇 개로 학습해 평균낼지")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="신경망 학습 장치 (기본: CUDA가 있으면 GPU)")
    ap.add_argument(
        "--train",
        nargs="+",
        choices=["all", *TRAIN_MODEL_ORDER],
        default=["all"],
        metavar="MODEL",
        help="학습할 모델: all, rf, rf-lag, lightgbm, lstm, tsmixer (기본: all)",
    )
    ap.add_argument("--skip-nn", action="store_true",
                    help="호환 옵션: --train all에서 LSTM/TSMixer 제외")
    ap.add_argument("--output-dir", default=BASE_DIR / "compare_runs", type=Path,
                    help="실행 기록을 저장할 상위 폴더")
    args = ap.parse_args()
    if args.window < 1:
        ap.error("--window는 1 이상이어야 합니다.")
    if args.seeds < 1:
        ap.error("--seeds는 1 이상이어야 합니다.")
    if args.train_start_year >= args.val_year:
        ap.error("--train-start-year는 --val-year보다 앞서야 합니다.")
    if args.val_year >= args.test_year:
        ap.error("--val-year는 --test-year보다 앞서야 합니다.")
    if "all" in args.train and len(args.train) > 1:
        ap.error("--train all은 다른 모델 이름과 함께 쓸 수 없습니다.")
    if args.skip_nn and any(name in {"lstm", "tsmixer"} for name in args.train):
        ap.error("--skip-nn과 신경망 모델 선택을 함께 쓸 수 없습니다.")
    requested = set(TRAIN_MODEL_ORDER if args.train == ["all"] else args.train)
    if args.skip_nn:
        requested -= {"lstm", "tsmixer"}
    args.train_models = [name for name in TRAIN_MODEL_ORDER if name in requested]
    if not args.train_models:
        ap.error("학습할 모델이 하나 이상 필요합니다.")

    started = datetime.now().astimezone()
    run_id = started.strftime("%Y%m%d_%H%M%S_%f")[:-3] + f"_y{args.test_year}_w{args.window}_s{args.seeds}"
    output_root = args.output_dir.resolve()
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "run_id": run_id,
        "status": "running",
        "started_at": started.isoformat(timespec="seconds"),
        "settings": {
            "data": str(args.data.resolve()),
            "train_start_year": args.train_start_year,
            "test_year": args.test_year,
            "val_year": args.val_year,
            "window": args.window,
            "seeds": args.seeds,
            "device": args.device,
            "skip_nn": args.skip_nn,
            "train_models": args.train_models,
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
    write_json(run_dir / "metadata.json", metadata)

    log_path = run_dir / "console.log"
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
            with contextlib.redirect_stdout(Tee(sys.stdout, log_file)), contextlib.redirect_stderr(Tee(sys.stderr, log_file)):
                print("=" * 72)
                print("캠퍼스 보정 모델 비교")
                print(f"실행 ID : {run_id}")
                print(f"기록 폴더: {run_dir}")
                print("=" * 72)
                summary, seasons, predictions, n_train, n_val, n_test = run_comparison(args)
                save_results(run_dir, output_root, run_id, args, summary, seasons, predictions)
                print("\n저장 완료")
                print(f"  요약       {run_dir / 'metrics.csv'}")
                print(f"  계절별     {run_dir / 'season_metrics.csv'}")
                print(f"  시간별 예측 {run_dir / 'predictions.csv'}")
                print(f"  전체 로그   {log_path}")
                print(f"  누적 기록   {output_root / 'history.csv'}")
        metadata.update({
            "status": "completed",
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "rows": {"train": n_train, "validation": n_val, "test": n_test},
            "files": ["metadata.json", "console.log", "metrics.csv", "season_metrics.csv", "predictions.csv"],
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
        write_json(run_dir / "metadata.json", metadata)

if __name__ == "__main__":
    main()
