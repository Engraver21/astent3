"""纯几何的单元测试（标准库 unittest，不需要额外依赖）。"""

import cv2
import unittest

import numpy as np

from stentkit import geometry


class GeometryTests(unittest.TestCase):
    def test_oriented_geometry_rectangle(self):
        mask = np.zeros((120, 120), np.uint8)
        mask[50:60, 20:80] = 255                      # 60 长 10 宽的水平矩形
        _, labels = cv2.connectedComponents(mask)
        points = geometry.component_points(labels, 1)
        geom = geometry.oriented_geometry(points)
        self.assertAlmostEqual(geom.length, 60, delta=1.5)
        self.assertAlmostEqual(geom.width, 10, delta=1.5)
        self.assertAlmostEqual(geom.ratio, 6, delta=0.6)
        self.assertLess(geometry.angle_difference(geom.angle, 0), 2)

    def test_oriented_geometry_diagonal(self):
        mask = np.zeros((120, 120), np.uint8)
        for i in range(60):
            mask[20 + i, 20 + i] = 255
            mask[20 + i, 21 + i] = 255
        _, labels = cv2.connectedComponents(mask)
        geom = geometry.oriented_geometry(geometry.component_points(labels, 1))
        self.assertAlmostEqual(geom.angle, 45, delta=3)
        self.assertGreater(geom.ratio, 20)

    def test_box_iou(self):
        self.assertAlmostEqual(geometry.box_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(geometry.box_iou((0, 0, 10, 10), (20, 20, 10, 10)), 0.0)
        self.assertAlmostEqual(geometry.box_iou((0, 0, 10, 10), (5, 0, 10, 10)), 1 / 3, delta=1e-6)

    def test_polygon_min_gap(self):
        a = np.array([[0, 0], [10, 0]], np.float32)
        b = np.array([[14, 0], [20, 0]], np.float32)
        self.assertAlmostEqual(geometry.polygon_min_gap(a, b), 4.0, delta=1e-6)

    def test_angle_difference(self):
        self.assertAlmostEqual(geometry.angle_difference(10, 30), 20)
        self.assertAlmostEqual(geometry.angle_difference(170, 10), 20)
        self.assertAlmostEqual(geometry.angle_difference(0, 90), 90)

    def test_fill_ratio(self):
        ink = np.zeros((50, 50), np.uint8)
        ink[10:20, 10:30] = 255
        box = np.int32([[10, 10], [30, 10], [30, 20], [10, 20]])
        # fillPoly 的栅格化会把边界算进去，因此比值略小于 1，这里只验证语义
        self.assertGreater(geometry.fill_ratio(ink, box), 0.8)
        self.assertLessEqual(geometry.fill_ratio(ink, box), 1.0)
        empty = np.int32([[0, 0], [5, 0], [5, 5], [0, 5]])
        self.assertAlmostEqual(geometry.fill_ratio(ink, empty), 0.0, delta=0.02)


if __name__ == "__main__":
    unittest.main()
