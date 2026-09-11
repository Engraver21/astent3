"""尺度估计：先定"本图一个支架有多大"，后面一切都跟着它缩放。

为什么这样做：同一堆支架在不同高度/角度拍出来，杆长会差 2~4 倍
（实测样张中位杆长 35~183 px）。V2 用固定像素常数（黑帽核 31、面积下限 35、
最小长度 12），因此只在某一档尺度上勉强可用；V1 虽然有自适应的长度窗口，
但核尺寸仍是固定思路。V3 用核尺寸扫描 + 平台法把 L（典型杆长）和 W（典型杆宽）估出来。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from . import enhance, geometry
from .config import DetectConfig


@dataclass
class KernelRow:
    kernel: int
    count: int
    length: float
    width: float
    otsu: float


@dataclass
class ScaleEstimate:
    kernel: int
    length: float
    width: float
    rows: List[KernelRow] = field(default_factory=list)
    note: str = ""


def _relative_change(a: int, b: int) -> float:
    largest = max(a, b, 1)
    return abs(a - b) / largest


def _plateau(counts: List[int], tolerance: float) -> tuple:
    """找候选数随核变化最平缓的一段，返回 (start, end)；找不到就返回 None。"""
    best = None
    start = 0
    for i in range(len(counts) - 1):
        if _relative_change(counts[i], counts[i + 1]) <= tolerance:
            if start is None:
                start = i
        else:
            if start is not None and i - start >= 1:
                span = i - start
                if best is None or span > best[1] - best[0] or (
                    span == best[1] - best[0] and counts[i] > counts[best[1]]
                ):
                    best = (start, i)
            start = None
    if start is not None and len(counts) - 1 - start >= 1:
        span = len(counts) - 1 - start
        if best is None or span > best[1] - best[0]:
            best = (start, len(counts) - 1)
    return best


def _component_stats(mask, aspect_min: float, area_min: float) -> tuple:
    """返回 (数量, 长度列表, 宽度列表)。"""
    count, labels, stats, _ = enhance.connected_components(mask)
    lengths, widths = [], []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < area_min:
            continue
        geom = geometry.component_geometry(labels, label, (x, y, w, h))
        if geom is None or geom.ratio < aspect_min:
            continue
        lengths.append(geom.length)
        widths.append(geom.width)
    return len(lengths), lengths, widths


def estimate_scale(gray_eq: np.ndarray, config: DetectConfig) -> ScaleEstimate:
    rows: List[KernelRow] = []
    for kernel in config.kernel_sweep:
        response = enhance.blackhat(gray_eq, kernel)
        otsu = enhance.otsu_value(response, positive_only=True)
        mask = enhance.threshold_mask(response, otsu, config.close_kernel)
        area_min = config.scale_area_ratio * kernel * kernel
        count, lengths, widths = _component_stats(mask, config.scale_aspect_min, area_min)
        rows.append(KernelRow(
            kernel=kernel,
            count=count,
            length=float(np.median(lengths)) if lengths else 0.0,
            width=float(np.median(widths)) if widths else 0.0,
            otsu=otsu,
        ))

    counts = [row.count for row in rows]
    note = ""
    span = _plateau(counts, config.plateau_tolerance)
    if span is None:
        # 没有平台：取"计数处于中位"的那一档，避免踩到噪声爆炸(小核)或目标消失(大核)
        order = sorted(range(len(rows)), key=lambda i: counts[i])
        index = order[len(order) // 2]
        note = "未找到稳定平台，退化取中位候选数的核"
    else:
        index = (span[0] + span[1]) // 2
        note = f"核平台 k={rows[span[0]].kernel}~{rows[span[1]].kernel}"

    chosen = rows[index]
    length = chosen.length if chosen.length > 0 else 40.0
    width = chosen.width if chosen.width > 0 else max(3.0, length / 12.0)
    return ScaleEstimate(kernel=chosen.kernel, length=length, width=width, rows=rows, note=note)


def size_window(scale: ScaleEstimate, config: DetectConfig) -> dict:
    """由尺度得到的几何筛选窗口（全部是相对量）。"""
    return {
        "length_min": config.length_min_ratio * scale.length,
        "length_max": config.length_max_ratio * scale.length,
        "width_min": config.width_min_ratio * scale.width,
        "width_max": config.width_max_ratio * scale.width,
        "area_min": config.area_min_ratio * scale.length * scale.width,
    }
