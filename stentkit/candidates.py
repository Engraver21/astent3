"""候选提取：连通域筛选、归并、（可选且默认关闭的）线段兜底。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from . import enhance, geometry
from .config import DetectConfig
from .scale import ScaleEstimate, size_window


@dataclass
class Candidate:
    """一个候选支架（或支架的一段）。"""

    x: int
    y: int
    w: int
    h: int
    length: float
    width: float
    ratio: float
    area: int
    box: np.ndarray                  # (4,2) 角点
    angle: float
    source: str = "component"        # component / merged / hough
    fill: float = 0.0                # 框内着墨率（置信度）
    support: float = 0.0             # 跨阈值支持度（置信度，不参与删除）
    mean_response: float = 0.0
    parts: int = 1
    label: int = 0

    @property
    def bbox(self) -> tuple:
        return self.x, self.y, self.w, self.h

    def to_dict(self) -> dict:
        return {
            "x": self.x, "y": self.y, "w": self.w, "h": self.h,
            "length": round(self.length, 2), "width": round(self.width, 2),
            "ratio": round(self.ratio, 2), "area": self.area,
            "angle": round(self.angle, 2), "source": self.source,
            "fill": round(self.fill, 4), "support": self.support,
            "parts": self.parts,
            "box": [[int(px), int(py)] for px, py in self.box],
        }


def candidate_from_label(labels: np.ndarray, response: np.ndarray, label: int,
                         stats: np.ndarray) -> Optional[Candidate]:
    x, y, w, h, area = stats[label]
    points = geometry.component_points(labels, label, (x, y, w, h))
    geom = geometry.oriented_geometry(points)
    if geom is None:
        return None
    values = response[y:y + h, x:x + w][labels[y:y + h, x:x + w] == label]
    return Candidate(
        x=int(x), y=int(y), w=int(w), h=int(h),
        length=geom.length, width=geom.width, ratio=geom.ratio, area=int(area),
        box=geom.box, angle=geom.angle, source="component",
        mean_response=float(values.mean()) if values.size else 0.0,
        label=label,
    )


def extract_components(mask: np.ndarray, response: np.ndarray, scale: ScaleEstimate,
                       config: DetectConfig, *, window: Optional[dict] = None,
                       ink_mask: Optional[np.ndarray] = None) -> List[Candidate]:
    """按尺度窗口筛选连通域候选。"""
    bounds = window or size_window(scale, config)
    count, labels, stats, _ = enhance.connected_components(mask)
    candidates: List[Candidate] = []
    for label in range(1, count):
        candidate = candidate_from_label(labels, response, label, stats)
        if candidate is None:
            continue
        if candidate.area < bounds["area_min"]:
            continue
        if candidate.ratio < config.aspect_min:
            continue
        if not (bounds["length_min"] <= candidate.length <= bounds["length_max"]):
            continue
        if candidate.width < bounds["width_min"] or candidate.width > bounds["width_max"]:
            continue
        if ink_mask is not None:
            candidate.fill = geometry.fill_ratio(ink_mask, candidate.box)
        candidates.append(candidate)
    return candidates


def _merge_members(members: Sequence[Candidate]) -> Candidate:
    """把若干候选段合成一个对象（几何用角点并集的最小外接矩形）。"""
    points = np.vstack([member.box for member in members]).astype(np.float32)
    geom = geometry.oriented_geometry(points.reshape(-1, 1, 2))
    xs = [member.x for member in members]
    ys = [member.y for member in members]
    xe = [member.x + member.w for member in members]
    ye = [member.y + member.h for member in members]
    if geom is None:
        geom = geometry.oriented_geometry(np.array(
            [[min(xs), min(ys)], [max(xe), min(ys)], [max(xe), max(ye)], [min(xs), max(ye)]],
            dtype=np.float32).reshape(-1, 1, 2))
    return Candidate(
        x=min(xs), y=min(ys), w=max(xe) - min(xs), h=max(ye) - min(ys),
        length=geom.length, width=geom.width, ratio=geom.ratio,
        area=int(sum(member.area for member in members)),
        box=geom.box, angle=geom.angle, source="merged",
        fill=float(np.mean([member.fill for member in members])),
        support=float(np.mean([member.support for member in members])),
        mean_response=float(np.mean([member.mean_response for member in members])),
        parts=sum(member.parts for member in members),
    )


def merge_when_parallel(candidates: Sequence[Candidate], image_shape,
                        config: DetectConfig) -> List[Candidate]:
    """目标很少且方向高度一致时，整体并成一个对象。

    近景单支架样张上，一个支架常被分成 2~5 段近乎平行的长条；
    此时"候选少 + 方向一致 + 合起来的外接框不算大"就是同一个支架的特征。
    """
    if len(candidates) < 2 or not config.single_scene_merge:
        return list(candidates)
    angles = [item.angle for item in candidates]
    if max(angles) - min(angles) > config.group_angle_tolerance:
        return list(candidates)
    xs = [item.x for item in candidates]
    ys = [item.y for item in candidates]
    xe = [item.x + item.w for item in candidates]
    ye = [item.y + item.h for item in candidates]
    image_area = float(image_shape[0] * image_shape[1])
    union_area = float((max(xe) - min(xs)) * (max(ye) - min(ys)))
    if image_area <= 0 or union_area / image_area > config.single_scene_area_limit:
        return list(candidates)
    return [_merge_members(candidates)]


def group_candidates(candidates: Sequence[Candidate], scale: ScaleEstimate,
                     config: DetectConfig, *, gap_ratio: Optional[float] = None,
                     enforce_length_limit: bool = True) -> List[Candidate]:
    """把同一支架的若干段并成一个对象。

    规则：两段之间间隙 < gap_ratio * L，且长轴夹角 < angle_tolerance 才归并。
    V1 在"只有一个支架"的样张上给出 2 个框，就是缺了这一步（同一个支架的两段各算一个）。
    """
    if len(candidates) < 2:
        return list(candidates)

    gap_limit = (config.group_gap_ratio if gap_ratio is None else gap_ratio) * scale.length
    length_limit = config.length_max_ratio * scale.length
    parent = list(range(len(candidates)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if geometry.angle_difference(candidates[i].angle, candidates[j].angle) > config.group_angle_tolerance:
                continue
            if geometry.polygon_min_gap(candidates[i].box, candidates[j].box) > gap_limit:
                continue
            if enforce_length_limit:
                # 归并后的整体长度仍须落在尺寸窗口内，否则说明是"相邻的两个支架"而不是"同一支架的两段"
                if max(candidates[i].length, candidates[j].length) > length_limit:
                    continue
                if (candidates[i].length ** 2 + candidates[j].length ** 2) ** 0.5 > length_limit:
                    continue
            union(i, j)

    groups: Dict[int, List[Candidate]] = {}
    for index, candidate in enumerate(candidates):
        groups.setdefault(find(index), []).append(candidate)

    merged: List[Candidate] = []
    for members in groups.values():
        if len(members) == 1:
            merged.append(members[0])
            continue
        merged.append(_merge_members(members))
    return merged


def hough_candidates(mask: np.ndarray, scale: ScaleEstimate, config: DetectConfig) -> List[Candidate]:
    """线段兜底——默认关闭。

    V2 的教训：在纹理背景上 HoughLinesP 会检出成千上万条线段，
    这种"兜底"会把结果从 26 个变成 120 个。这里只保留一个被严格门控的实现：
    线段必须有足够比例落在掩膜上、总数量有上限、且长度接近典型杆长。
    另外注意 OpenCV 5 的返回值形状是 (N,4)（4.x 是 (N,1,4)），统一 reshape 处理。
    """
    if not config.use_hough_fallback:
        return []
    edges = cv2.Canny(mask, 30, 100)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 18,
        minLineLength=max(12, int(scale.length * 0.5)),
        maxLineGap=max(2, int(scale.length * 0.08)),
    )
    if lines is None:
        return []
    lines = np.asarray(lines).reshape(-1, 4)      # 兼容 OpenCV 4/5
    found: List[Candidate] = []
    for x1, y1, x2, y2 in lines:
        length = float(np.hypot(x2 - x1, y2 - y1))
        if not (scale.length * 0.6 <= length <= scale.length * 1.6):
            continue
        width = max(3.0, min(scale.width * 1.5, scale.length * 0.2))
        x, y, w, h = geometry.line_box(x1, y1, x2, y2, width)
        box = np.int32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        fill = geometry.fill_ratio(mask, box)
        if fill < config.hough_min_fill:
            continue
        geom = geometry.oriented_geometry(box.reshape(-1, 1, 2))
        if geom is None:
            continue
        found.append(Candidate(
            x=int(x), y=int(y), w=int(w), h=int(h),
            length=geom.length, width=geom.width, ratio=geom.ratio,
            area=int(w * h * fill), box=geom.box, angle=geom.angle,
            source="hough", fill=fill,
        ))
        if len(found) >= config.hough_max_boxes:
            break
    return found


def merge_duplicates(candidates: Sequence[Candidate], scale: ScaleEstimate,
                     config: DetectConfig) -> List[Candidate]:
    """去掉高度重叠的重复框（保留着墨率更高、更完整的那个）。"""
    ordered = sorted(candidates, key=lambda c: (c.parts, c.length, c.fill), reverse=True)
    kept: List[Candidate] = []
    for candidate in ordered:
        if any(geometry.box_iou(candidate.bbox, other.bbox) >= 0.6 for other in kept):
            continue
        kept.append(candidate)
    return kept
