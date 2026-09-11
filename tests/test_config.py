"""配置读写测试。"""

import json
import tempfile
import unittest
from pathlib import Path

from stentkit import DetectConfig, load_config


class ConfigTests(unittest.TestCase):
    def test_roundtrip(self):
        config = DetectConfig()
        restored = DetectConfig.from_dict(config.to_dict())
        self.assertEqual(restored.kernel_sweep, config.kernel_sweep)
        self.assertEqual(restored.threshold_ratios, config.threshold_ratios)
        self.assertEqual(restored.length_min_ratio, config.length_min_ratio)

    def test_load_with_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"group_enabled": False, "length_min_ratio": 0.5}),
                            encoding="utf-8")
            config = load_config(path, {"length_min_ratio": 0.7})
            self.assertFalse(config.group_enabled)
            self.assertAlmostEqual(config.length_min_ratio, 0.7)


if __name__ == "__main__":
    unittest.main()

