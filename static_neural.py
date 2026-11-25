"""브라우저로 내보낸 LSTM/TSMixer를 NumPy로 재현하는 작은 추론기."""
from __future__ import annotations

import math

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def _gelu(x):
    values = np.asarray(x, dtype=float)
    erf = np.vectorize(math.erf, otypes=[float])
    return 0.5 * values * (1.0 + erf(values / math.sqrt(2.0)))


def _linear(state, prefix, x):
    weight = np.asarray(state[f"{prefix}.weight"], dtype=float)
    bias = np.asarray(state[f"{prefix}.bias"], dtype=float)
    return np.asarray(x, dtype=float) @ weight.T + bias


def _layer_norm(state, prefix, x, eps=1e-5):
    values = np.asarray(x, dtype=float)
    mean = values.mean(axis=-1, keepdims=True)
    variance = ((values - mean) ** 2).mean(axis=-1, keepdims=True)
    weight = np.asarray(state[f"{prefix}.weight"], dtype=float)
    bias = np.asarray(state[f"{prefix}.bias"], dtype=float)
    return (values - mean) / np.sqrt(variance + eps) * weight + bias


def _lstm(payload, sequence):
    state = payload["state"]
    layer_input = np.asarray(sequence, dtype=float)
    layers = int(payload.get("layers", 2))
    for layer in range(layers):
        hidden = len(state[f"lstm.weight_hh_l{layer}"][0])
        h = np.zeros(hidden, dtype=float)
        c = np.zeros(hidden, dtype=float)
        outputs = []
        w_ih = np.asarray(state[f"lstm.weight_ih_l{layer}"], dtype=float)
        w_hh = np.asarray(state[f"lstm.weight_hh_l{layer}"], dtype=float)
        bias = np.asarray(state[f"lstm.bias_ih_l{layer}"], dtype=float) + np.asarray(
            state[f"lstm.bias_hh_l{layer}"], dtype=float
        )
        for x in layer_input:
            gates = w_ih @ x + w_hh @ h + bias
            i, f, g, o = np.split(gates, 4)
            i, f, o = _sigmoid(i), _sigmoid(f), _sigmoid(o)
            c = f * c + i * np.tanh(g)
            h = o * np.tanh(c)
            outputs.append(h.copy())
        layer_input = np.asarray(outputs)
    joined = np.concatenate([layer_input[-1], np.asarray(sequence)[-1]])
    return _linear(state, "head.2", _gelu(_linear(state, "head.0", joined)))


def _tsmixer(payload, sequence):
    state = payload["state"]
    original = np.asarray(sequence, dtype=float)
    z = _linear(state, "proj", original)
    blocks = int(payload.get("blocks", 3))
    for block in range(blocks):
        prefix = f"blocks.{block}"
        temporal = _layer_norm(state, f"{prefix}.norm_t", z).T
        temporal = _gelu(_linear(state, f"{prefix}.time.0", temporal)).T
        z = z + temporal
        feature = _layer_norm(state, f"{prefix}.norm_f", z)
        feature = _gelu(_linear(state, f"{prefix}.feat.0", feature))
        z = z + _linear(state, f"{prefix}.feat.3", feature)
    joined = np.concatenate([z.reshape(-1), original[-1]])
    return _linear(state, "head.2", _gelu(_linear(state, "head.0", joined)))


def neural_delta(payload: dict, sequence) -> np.ndarray:
    """원 입력 시퀀스를 받아 세 목표의 보정량을 반환한다."""
    values = np.asarray(sequence, dtype=float)
    normalized = (
        values - np.asarray(payload["input_mean"], dtype=float)
    ) / np.asarray(payload["input_std"], dtype=float)
    scaled = _lstm(payload, normalized) if payload["kind"] == "lstm" else _tsmixer(payload, normalized)
    return scaled * np.asarray(payload["target_std"], dtype=float) + np.asarray(
        payload["target_mean"], dtype=float
    )
