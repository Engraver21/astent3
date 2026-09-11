"""诊断：某个阈值下，候选是被哪一条规则筛掉的。"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

from . import enhance, geometry
from .config import DetectConfig
from .scale import ScaleEstimate, size_window


def explain_components(mask: np.ndarray, scale: ScaleEstimate, config: DetectConfig,
                       *, window: Optional[dict] = None) -> dict:
    bounds = window or size_window(scale, config)
    count, labels, stats, _ = enhance.connected_components(mask)
    reasons = Counter()
    examples = {"长度超范围": [], "宽度超范围": [], "长宽比不足": [], "面积不足": []}
    kept_lengths, kept_widths = [], []

    for label in range(1, count):
        x, y, w, h, area = stats[label]
        geom = geometry.component_geometry(labels, label, (x, y, w, h))
        if geom is None:
            reasons["几何退化"] += 1
            continue
        if area < bounds["area_min"]:
            reasons["面积不足"] += 1
            examples["面积不足"].append((geom.length, geom.width, int(area)))
            continue
        if geom.ratio < config.aspect_min:
            reasons["长宽比不足"] += 1
            examples["长宽比不足"].append((geom.length, geom.width, int(area)))
            continue
        if not (bounds["length_min"] <= geom.length <= bounds["length_max"]):
            reasons["长度超范围"] += 1
            examples["长度超范围"].append((geom.length, geom.width, int(area)))
            continue
        if not (bounds["width_min"] <= geom.width <= bounds["width_max"]):
            reasons["宽度超范围"] += 1
            examples["宽度超范围"].append((geom.length, geom.width, int(area)))
            continue
        reasons["通过"] += 1
        kept_lengths.append(geom.length)
        kept_widths.append(geom.width)

    return {
        "连通域总数": count - 1,
        "窗口": {k: round(v, 2) for k, v in bounds.items()},
        "筛除统计": dict(reasons),
        "通过者的长度": (round(float(np.median(kept_lengths)), 1) if kept_lengths else None),
        "通过者的宽度": (round(float(np.median(kept_widths)), 1) if kept_widths else None),
        "被丢样本": {k: [(round(a, 1), round(b, 1), c) for a, b, c in v[:8]] for k, v in examples.items()},
    }
