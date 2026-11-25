from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from compare_models import SEQ_FEATURES, build_nets
from static_neural import neural_delta
from web_export import export_neural_model


ROOT = Path(__file__).resolve().parents[1]


class StaticNeuralParityTest(unittest.TestCase):
    def check_model(self, key: str):
        torch.manual_seed(7)
        rng = np.random.default_rng(7)
        window = 6
        channels = len(SEQ_FEATURES)
        LSTMNet, TSMixer = build_nets()
        net = LSTMNet(channels) if key == "lstm" else TSMixer(window, channels)
        net.eval()
        raw = rng.normal(size=(window, channels)).astype(np.float32)
        input_mean = rng.normal(size=channels).astype(np.float32)
        input_std = rng.uniform(0.5, 2.0, size=channels).astype(np.float32)
        target_mean = rng.normal(size=3).astype(np.float32)
        target_std = rng.uniform(0.5, 2.0, size=3).astype(np.float32)
        normalized = (raw - input_mean) / input_std
        with torch.no_grad():
            expected = net(torch.from_numpy(normalized[None])).numpy()[0] * target_std + target_mean

        checkpoint = {
            "model": key,
            "state_dict": net.state_dict(),
            "window": window,
            "sequence_features": SEQ_FEATURES,
            "input_mean": input_mean,
            "input_std": input_std,
            "target_mean": target_mean,
            "target_std": target_std,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            torch.save(checkpoint, path)
            payload = export_neural_model(path, key)

        actual = neural_delta(payload, raw)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

        if shutil.which("node"):
            node_program = r"""
const fs = require('fs');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
global.window = {ONULCHUPNA: {config: {}, campus_model: input.payload}};
const source = fs.readFileSync(input.script, 'utf8');
eval(source + '\n;globalThis.__answer = Standalone._neuralDelta(window.ONULCHUPNA.campus_model, input.raw);');
process.stdout.write(JSON.stringify(globalThis.__answer));
"""
            completed = subprocess.run(
                ["node", "-e", node_program],
                input=json.dumps({
                    "script": str(ROOT / "docs" / "standalone.js"),
                    "payload": payload,
                    "raw": raw.tolist(),
                }),
                text=True,
                capture_output=True,
                check=True,
            )
            browser = np.asarray(json.loads(completed.stdout), dtype=float)
            np.testing.assert_allclose(browser, expected, rtol=2e-4, atol=2e-4)

    def test_lstm(self):
        self.check_model("lstm")

    def test_tsmixer(self):
        self.check_model("tsmixer")


if __name__ == "__main__":
    unittest.main()
