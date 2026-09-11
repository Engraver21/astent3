"""端到端冒烟测试：造一张已知数量的合成图，验证流程能数对。"""

import unittest

import cv2
import numpy as np

from stentkit import DetectConfig, detect_image


def synthetic(count=3, length=64, width=8, size=None) -> np.ndarray:
    if size is None:
        # 杆长 64、倾斜 ±25° 时横向约占 60px，间距给足 95px 才能保证互不相连
        size = (max(360, 95 * count + 60), 280)
    height, width_px = size[1], size[0]
    image = np.full((height, width_px, 3), 235, np.uint8)
    margin = 40
    step = (width_px - 2 * margin) // max(1, count)
    for index in range(count):
        cx = margin + step * index + step // 2
        cy = height // 2 + (index % 2) * 30 - 15
        angle = np.deg2rad(-25 + index * 25)
        dx = int(np.cos(angle) * length / 2)
        dy = int(np.sin(angle) * length / 2)
        cv2.line(image, (cx - dx, cy - dy), (cx + dx, cy + dy), (60, 60, 60), width)
    noise = np.random.default_rng(0).normal(0, 3, image.shape).astype(np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def draw_rod(image, cx, cy, length, width, angle_deg, color=(60, 60, 60)):
    angle = np.deg2rad(angle_deg)
    dx = int(np.cos(angle) * length / 2)
    dy = int(np.sin(angle) * length / 2)
    cv2.line(image, (cx - dx, cy - dy), (cx + dx, cy + dy), color, width)


def scene_touching_pair() -> np.ndarray:
    """一根独立 + 两根首尾几乎贴在一起（中间只留 3px）。真值 3。"""
    image = np.full((300, 520, 3), 235, np.uint8)
    draw_rod(image, 90, 150, 64, 8, 0)                 # 独立的一根
    draw_rod(image, 270, 150, 64, 8, 0)                # 与下一根贴住
    draw_rod(image, 270 + 67, 150, 64, 8, 0)
    noise = np.random.default_rng(1).normal(0, 3, image.shape).astype(np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def scene_highlight_broken() -> np.ndarray:
    """三根完整 + 一根中间被高光打断成两段（缺 10px）。真值 4。"""
    image = np.full((340, 720, 3), 235, np.uint8)
    draw_rod(image, 80, 170, 64, 8, 0)                  # 完整
    draw_rod(image, 220, 170, 64, 8, 0)                 # 完整
    draw_rod(image, 360, 170, 64, 8, 0)                 # 完整
    draw_rod(image, 530, 170, 27, 8, 0)                 # 高光打断：左半段
    draw_rod(image, 530 + 27 + 10, 170, 27, 8, 0)       # 高光打断：右半段（中间缺 10px）
    noise = np.random.default_rng(2).normal(0, 3, image.shape).astype(np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


class PipelineTests(unittest.TestCase):
    def test_counts_synthetic_rods(self):
        for count in (1, 3, 5):
            with self.subTest(count=count):
                result = detect_image(synthetic(count=count), DetectConfig())
                self.assertEqual(result.count, count)

    def test_scale_is_estimated(self):
        result = detect_image(synthetic(count=3), DetectConfig())
        self.assertGreater(result.scale.length, 40)
        self.assertLess(result.scale.length, 90)
        self.assertGreater(result.scale.width, 3)
        self.assertLess(result.scale.width, 20)

    def test_group_merges_touching_pieces(self):
        image = synthetic(count=1, length=90, width=8, size=(260, 240))
        config = DetectConfig()
        with_group = detect_image(image, config).count
        without_group = detect_image(image, DetectConfig(group_enabled=False)).count
        self.assertLessEqual(with_group, without_group)

    def test_touching_rods_are_counted_separately(self):
        """贴在一起的支架要拆开数（问题 1）。"""
        result = detect_image(scene_touching_pair(), DetectConfig())
        self.assertEqual(result.count, 3)

    def test_highlight_broken_rod_counts_as_one(self):
        """被高光打断的支架要拼回一个（问题 2）。"""
        result = detect_image(scene_highlight_broken(), DetectConfig())
        self.assertEqual(result.count, 4)

    def test_single_closeup_can_merge_with_flag(self):
        """单支近景（整张图就一支）：打开开关后才整体并成 1 个。"""
        image = scene_highlight_broken()
        merged = detect_image(image, DetectConfig(single_scene_merge=True)).count
        default = detect_image(image, DetectConfig()).count
        self.assertEqual(default, 4)
        self.assertLess(merged, default)

    def test_numbering_covers_every_object(self):
        """每个识别到的支架都要有独立序号（问题 3）。"""
        from stentkit import pipeline as pipeline_mod

        result = detect_image(synthetic(count=4), DetectConfig())
        canvas = pipeline_mod.draw_result(result)
        self.assertEqual(canvas.shape, result.image.shape)
        self.assertEqual(len(result.objects), result.count)
        self.assertEqual([item.index_in_group for item in result.objects],
                         [1] * result.count)


if __name__ == "__main__":
    unittest.main()
