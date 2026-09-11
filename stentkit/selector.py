"""阈值选择：不奖励「候选多」，只奖励「与尺度先验一致且稳定」。

V1/V2 都用「候选数最多的阈值」当选最佳（max(ranked)），而黑帽/反二值掩膜天然是
「阈值越高越碎、碎片越多」，于是选择规则会主动挑最碎的阈值：
实测 stents5 在 12 个候选阈值里原始候选数从 26 一路涨到 124，V2 正好选了最大的那一档。

V3 的做法：先按尺度窗口把候选限制成「像支架的」，再要求该阈值在相邻阈值上稳定，
最后才比数量；跨阈值支持度只写进结果当置信度，从不参与删除。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from . import candidates as cand_mod
from . import enhance, geometry
from .config import DetectConfig
from .scale import ScaleEstimate, size_window


@dataclass
class ThresholdRow:
    threshold: int
    ratio: float
    sized_count: int          # 通过尺度窗口的候选数
    total_count: int          # 全部细长连通域数（不筛尺寸）
    median_length: float
    mean_fill: float
    stability: float          # 与相邻阈值的一致性 [0,1]
    score: float

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "ratio": self.ratio,
            "sized_count": self.sized_count,
            "total_count": self.total_count,
            "median_length": round(self.median_length, 2),
            "mean_fill": round(self.mean_fill, 4),
            "stability": round(self.stability, 3),
            "score": round(self.score, 3),
        }


def threshold_candidates(base: float, config: DetectConfig) -> List[int]:
    values = {int(round(base * ratio)) for ratio in config.threshold_ratios}
    values.update(range(config.absolute_start, config.absolute_stop + 1, config.absolute_step))
    return sorted(value for value in values if 1 <= value <= 254)


def _relative_change(a: int, b: int) -> float:
    return abs(a - b) / max(a, b, 1)


def _stability(counts: Sequence[int], index: int, config: DetectConfig) -> float:
    """该阈值与前后各 span 个阈值的一致性：全在容差内记 1，否则按比例。"""
    span = config.stability_span
    neighbours = [counts[i] for i in range(max(0, index - span), min(len(counts), index + span + 1))
                  if i != index]
    if not neighbours:
        return 1.0
    good = sum(1 for value in neighbours
               if _relative_change(counts[index], value) <= config.stability_tolerance)
    return good / len(neighbours)


def _count_all(mask: np.ndarray) -> int:
    return int(enhance.connected_components(mask)[0] - 1)


def evaluate_thresholds(response: np.ndarray, scale: ScaleEstimate, ink_mask: np.ndarray,
                        config: DetectConfig) -> Tuple[int, List[ThresholdRow], list, list]:
    """扫描 Otsu 附近的阈值，返回 (最佳阈值, 扫描表, 该阈值下的候选, 全部阈值的结果)。"""
    base = enhance.otsu_value(response, positive_only=True)
    if base <= 0:
        base = float(np.percentile(response, 99))
    window = size_window(scale, config)
    grid = threshold_candidates(base, config)

    per_threshold = []
    counts: List[int] = []
    masks = {}
    for threshold in grid:
        mask = enhance.threshold_mask(response, threshold, config.close_kernel)
        masks[threshold] = mask
        found = cand_mod.extract_components(mask, response, scale, config,
                                            window=window, ink_mask=ink_mask)
        per_threshold.append((threshold, found))
        counts.append(len(found))

    rows: List[ThresholdRow] = []
    for index, (threshold, found) in enumerate(per_threshold):
        stability = _stability(counts, index, config)
        lengths = [item.length for item in found]
        fills = [item.fill for item in found]
        mean_fill = float(np.mean(fills)) if fills else 0.0
        # 打分：数量 × 稳定度 × 框内着墨率。这里的数量已经是"像支架的候选数"，
        # 不是原始细长连通域数，所以碎片化不再直接换成分数。
        score = float(len(found)) * (0.5 + 0.5 * stability) * (0.6 + 0.4 * mean_fill)
        rows.append(ThresholdRow(
            threshold=threshold,
            ratio=round(threshold / base, 3) if base else 0.0,
            sized_count=len(found),
            total_count=_count_all(masks[threshold]),
            median_length=float(np.median(lengths)) if lengths else 0.0,
            mean_fill=mean_fill,
            stability=stability,
            score=score,
        ))

    best_index = _pick_best(rows, base, config)
    return per_threshold[best_index][0], rows, per_threshold[best_index][1], per_threshold


def _pick_best(rows: Sequence[ThresholdRow], base: float, config: DetectConfig) -> int:
    """选"尺寸合格候选数的最高平台"的中点。

    这是 V3 和 V1/V2 最本质的区别：不是挑候选数最大的那一档（那等于奖励碎片化），
    而是先取候选数的最大值，再在"接近最大值的连续阈值区间"里取中点——
    极值点附近的平台说明这段分割是稳定的，而平台的端点往往是开始碎/开始粘的临界。
    """
    counts = [row.sized_count for row in rows]
    if not counts or max(counts) == 0:
        return max(range(len(rows)), key=lambda index: rows[index].score)

    best_count = max(counts)
    tolerance = config.stability_tolerance
    # 两道门槛：
    #   1) 数量接近最大值（平台）；
    #   2) 候选框内着墨率不能太低——噪声爆炸的阈值上，总有几个小碎片"尺寸合格"，
    #      但它们落在目标上的比例极低（实测 0% vs 正常 90%），用这一条把它们挡掉。
    eligible = [
        count >= best_count * (1 - tolerance) and row.mean_fill >= config.min_mean_fill
        for count, row in zip(counts, rows)
    ]

    runs = []
    start = None
    for index, ok in enumerate(eligible):
        if ok and start is None:
            start = index
        elif not ok and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(eligible) - 1))
    if not runs:
        return max(range(len(rows)), key=lambda index: rows[index].score)

    # 平台越宽越可信；同样宽时取着墨率更高的那段
    def run_key(run):
        span = run[1] - run[0]
        fill = float(np.mean([rows[i].mean_fill for i in range(run[0], run[1] + 1)]))
        return span, fill

    best_run = max(runs, key=run_key)
    # 在平台内部再按分数取一点，避免正好卡在平台的边缘
    middle = (best_run[0] + best_run[1]) // 2
    window = [i for i in range(best_run[0], best_run[1] + 1)]
    return max(window, key=lambda index: (rows[index].score, -abs(index - middle)))


def compute_support(per_threshold: Sequence[Tuple[int, list]], chosen: Sequence) -> None:
    """标注每个候选在多少个阈值上被支持（置信度，不参与删除）。"""
    box_sets = [[item.bbox for item in found] for _, found in per_threshold]
    for candidate in chosen:
        support = 0
        for box_set in box_sets:
            if any(geometry.box_iou(candidate.bbox, other) >= 0.3 for other in box_set):
                support += 1
        candidate.support = float(support)
