"""纯几何工具：最小外接矩形、IoU、多边形间隙、角度。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Geometry:
    """一个连通域的方向包围盒几何。"""

    length: float          # 长边
    width: float           # 短边
    ratio: float           # 长宽比
    box: np.ndarray        # (4,2) int32 角点
    center: Tuple[float, float]
    angle: float           # 长轴角度，单位度，范围 [0,180)


def oriented_geometry(points: Optional[np.ndarray]) -> Optional[Geometry]:
    """由点集计算方向包围盒几何。"""
    if points is None or len(points) == 0:
        return None
    rect = cv2.minAreaRect(points)
    (cx, cy), (bw, bh), raw_angle = rect
    if bw <= 0 or bh <= 0:
        return None
    length, width = max(bw, bh), min(bw, bh)
    box = np.int32(np.round(cv2.boxPoints(rect)))
    angle = raw_angle % 180.0
    if bw < bh:
        angle = (angle + 90.0) % 180.0
    return Geometry(length=length, width=width, ratio=length / width,
                    box=box, center=(float(cx), float(cy)), angle=float(angle))


def component_points(labels: np.ndarray, label: int, bbox: Optional[Sequence[int]] = None) -> Optional[np.ndarray]:
    """取某个 label 的像素点集。注意 labels 是 connectedComponents 输出的标签图（0,1,2...）。

    给了 bbox=(x, y, w, h) 就只在局部窗口里找，避免对整个大图做一次全图比较。
    """
    if bbox is None:
        return cv2.findNonZero(np.uint8(labels == label) * 255)
    x, y, w, h = (int(v) for v in bbox)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(labels.shape[1], x + w), min(labels.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = labels[y0:y1, x0:x1]
    local = cv2.findNonZero(np.uint8(patch == label) * 255)
    if local is None:
        return None
    return (local.astype(np.float32) + np.array([x0, y0], np.float32)).astype(np.float32)


def component_geometry(labels: np.ndarray, label: int,
                       bbox: Optional[Sequence[int]] = None) -> Optional[Geometry]:
    return oriented_geometry(component_points(labels, label, bbox))


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """两个 (x, y, w, h) 框的 IoU。"""
    x1, y1, w1, h1 = first[:4]
    x2, y2, w2, h2 = second[:4]
    left, top = max(x1, x2), max(y1, y2)
    right, bottom = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, right - left) * max(0, bottom - top)
    union = w1 * h1 + w2 * h2 - inter
    return float(inter / union) if union > 0 else 0.0


def polygon_min_gap(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """两个点集之间的最小距离（用角点近似，规模很小，够用）。"""
    a = np.asarray(points_a, dtype=np.float32).reshape(-1, 2)
    b = np.asarray(points_b, dtype=np.float32).reshape(-1, 2)
    if a.size == 0 or b.size == 0:
        return float("inf")
    diff = a[:, None, :] - b[None, :, :]
    return float(np.sqrt((diff ** 2).sum(axis=2)).min())


def angle_difference(angle_a: float, angle_b: float) -> float:
    """两条长轴的方向差，范围 [0,90]。"""
    delta = abs(angle_a - angle_b) % 180.0
    return float(min(delta, 180.0 - delta))


def fill_ratio(ink_mask: np.ndarray, box: np.ndarray) -> float:
    """方向包围盒内着墨像素占比（只作为置信度，不做删除依据）。"""
    box = np.int32(box)
    x, y, w, h = cv2.boundingRect(box)
    x0, y0 = max(0, x), max(0, y)
    x1 = min(ink_mask.shape[1], x + w)
    y1 = min(ink_mask.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    # 只在框的局部区域上栅格化，避免每次都分配整图大小的画布（大图上是 10 倍级差距）
    canvas = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(canvas, [box - np.array([x0, y0], dtype=np.int32)], 255)
    total = int(cv2.countNonZero(canvas))
    if total == 0:
        return 0.0
    patch = ink_mask[y0:y1, x0:x1]
    return float(cv2.countNonZero(cv2.bitwise_and(patch, canvas))) / total


def polygon_area(box: np.ndarray) -> float:
    return float(abs(cv2.contourArea(np.int32(box))))


def line_box(x1: float, y1: float, x2: float, y2: float, width: float) -> Tuple[float, float, float, float]:
    """把一条线段按给定宽度扩成外接矩形 (x, y, w, h)。"""
    half = max(1.0, width / 2.0)
    xs = [x1 - half, x2 - half, x2 + half, x1 + half]
    ys = [y1 - half, y2 - half, y2 + half, y1 + half]
    x, y = min(xs), min(ys)
    return x, y, max(xs) - x, max(ys) - y
