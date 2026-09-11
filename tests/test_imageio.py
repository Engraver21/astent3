"""图片读写：中文路径不应乱码，也不应静默失败。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from stentkit import imageio


class ImageIOTests(unittest.TestCase):
    def test_roundtrip_ascii(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.png"
            image = np.zeros((20, 30, 3), np.uint8)
            image[:, :] = (10, 20, 30)
            self.assertTrue(imageio.imwrite(path, image))
            loaded = imageio.imread(path)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.shape, image.shape)

    def test_roundtrip_chinese_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "01_原图-支架.png"
            image = np.full((15, 25, 3), 200, np.uint8)
            self.assertTrue(imageio.imwrite(path, image))
            self.assertTrue(path.exists())
            self.assertEqual(path.name, "01_原图-支架.png")   # 文件名没有被改写
            loaded = imageio.imread(path)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.shape, image.shape)

    def test_missing_file(self):
        self.assertIsNone(imageio.imread(Path("不存在的目录") / "没有这张图.png"))

    def test_jpeg_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "结果.jpg"
            image = np.full((16, 16, 3), 128, np.uint8)
            self.assertTrue(imageio.imwrite(path, image))
            self.assertIsNotNone(imageio.imread(path))


if __name__ == "__main__":
    unittest.main()

