"""成组与计数：把"粘在一起"和"被高光打断"这两种情况都数对。

单靠连通域计数会同时犯两个错：
  * 两个支架贴得近 → 粘成一块 → 长度超范围被整块丢掉（漏检）；
  * 支架中间有高光 → 一块被打断成两段 → 两段各自都太短（漏检）。

所以 V3 改成两步：
  1) 按距离把碎块聚成"组"（bridge_ratio * L 的膨胀半径）；
  2) 用"一个支架的典型面积 A0"算组里有几个：n = round(组面积 / A0)。
组并得大一些不影响结果，因为个数由面积决定，而不是由块数决定。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from . import enhance, geometry
from .config import DetectConfig
from .scale import ScaleEstimate


@dataclass
class Group:
    """一组空间上相邻的着墨块（可能是一个支架，也可能是几个粘在一起的支架）。"""

    pixels: np.ndarray            # (N, 2) float32，点的 (x, y)
    area: int                     # 组内着墨像素总数
    parts: int                    # 组内的连通块个数
    box: np.ndarray               # 方向包围盒
    length: float
    width: float
    angle: float


@dataclass
class StentObject:
    """一个被数出来的支架（带自己的框，方便画序号）。"""

    box: np.ndarray
    length: float
    width: float
    angle: float
    area: int
    group_area: int
    parts: int
    index_in_group: int
    count_in_group: int
    source: str = "group"

    @property
    def bbox(self) -> tuple:
        x, y, w, h = cv2.boundingRect(np.int32(self.box))
        return x, y, w, h

    def to_dict(self) -> dict:
        x, y, w, h = self.bbox
        return {
            "x": x, "y": y, "w": w, "h": h,
            "length": round(self.length, 2), "width": round(self.width, 2),
            "angle": round(self.angle, 2), "area": self.area,
            "group_area": self.group_area, "parts": self.parts,
            "index_in_group": self.index_in_group, "count_in_group": self.count_in_group,
            "source": self.source,
            "box": [[int(px), int(py)] for px, py in self.box],
        }


def material_filter(scale: ScaleEstimate, unit_area: float, config: DetectConfig):
    """返回"这块像素算不算支架材料"的判定函数。

    形状（长宽比、杆宽）按本图尺度严格卡住，只放宽长度上下限：
    两个支架粘在一起会让长度翻倍、高光打断会让长度腰斩，长度不能用来筛材料，
    但"细长、宽度和杆宽一致"这两条噪声块一般过不了。
    """
    min_width = config.width_min_ratio * scale.width * 0.8
    max_width = config.width_max_ratio * scale.width * 1.2
    min_length = config.fragment_length_min_ratio * scale.length
    max_length = config.fragment_length_max_ratio * scale.length
    min_area = config.fragment_area_ratio * max(1.0, unit_area)

    def accept(geom: geometry.Geometry, area: int) -> bool:
        return (
            geom.ratio >= config.aspect_min
            and min_width <= geom.width <= max_width
            and min_length <= geom.length <= max_length
            and area >= min_area
        )

    return accept


def build_groups(mask: np.ndarray, radius: int, min_area: int, accept=None) -> List[Group]:
    """把掩膜里的连通块按距离聚成组（accept 用来筛掉不成形的噪声块）。"""
    radius = max(1, int(radius))
    component_count, component_labels, stats, centroids = enhance.connected_components(mask)
    if component_count <= 1:
        return []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    dilated = cv2.dilate(mask, kernel)
    _, group_labels = cv2.connectedComponents(dilated, connectivity=8)

    # 每个连通块属于哪个组：取块内出现最多的膨胀标签（比用质心稳）
    buckets: Dict[int, List[int]] = {}
    for label in range(1, component_count):
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        if accept is not None:
            geom = geometry.component_geometry(component_labels, label, (x, y, w, h))
            if geom is None or not accept(geom, int(area)):
                continue
        patch = group_labels[y:y + h, x:x + w]
        inner = patch[component_labels[y:y + h, x:x + w] == label]
        inner = inner[inner > 0]
        if inner.size == 0:
            cx, cy = centroids[label]
            group_id = int(group_labels[int(round(cy)), int(round(cx))])
        else:
            group_id = int(np.bincount(inner).argmax())
        if group_id <= 0:
            continue
        buckets.setdefault(group_id, []).append(label)

    groups: List[Group] = []
    for group_id, labels in buckets.items():
        points = []
        area_total = 0
        for label in labels:
            x, y, w, h, area = stats[label]
            area_total += int(area)
            local = cv2.findNonZero(np.uint8(component_labels[y:y + h, x:x + w] == label) * 255)
            if local is not None:
                points.append(local.reshape(-1, 2) + np.array([x, y], np.float32))
        if not points:
            continue
        pixels = np.vstack(points).astype(np.float32)
        geom = geometry.oriented_geometry(pixels.reshape(-1, 1, 2))
        if geom is None:
            continue
        groups.append(Group(
            pixels=pixels, area=area_total, parts=len(labels), box=geom.box,
            length=geom.length, width=geom.width, angle=geom.angle,
        ))
    return groups


def estimate_unit_area(plausible_areas: Sequence[float], scale: ScaleEstimate,
                       config: DetectConfig) -> float:
    """估计"一个支架的典型着墨面积" A0。

    用"尺寸合格的单块"面积的**下半部分中位数**，而不是直接取中位数：
    粘连块（=2 个支架粘一起）的面积约为 2*A0，直接取中位数会被它拉高，
    于是两个粘在一起的支架会被算成 1 个。取下半部分即偏向"确定是单个"的那一批。
    这里假设照片里总有几处是摆得开的（用户实测也是如此），否则退化为尺度先验下限。
    """
    areas = [float(a) for a in plausible_areas if a > 0]
    fallback = max(1.0, config.unit_area_min_ratio * scale.length * scale.width)
    if not areas:
        return fallback
    areas.sort()
    mode = config.unit_area_mode
    if mode == "lower_half":
        subset = areas[:max(1, len(areas) // 2)] if len(areas) > 2 else areas[:1]
    elif mode == "p25":
        subset = [float(np.percentile(areas, 25))]
    elif mode == "trimmed":
        # 先按 25 分位估一个"单块"的量级，再把明显更大的（疑似 2 个粘一起的块）剔出统计
        base = float(np.percentile(areas, 25))
        subset = [value for value in areas if value <= base * config.unit_area_trim_factor] or areas
    elif mode == "mode":
        # 取面积分布的"众数峰"：一个支架的面积会形成一个尖峰，
        # 粘连块(≈2×)和高光残块(≈0.5×)只是两侧的尾巴。
        low, high = float(np.percentile(areas, 5)), float(np.percentile(areas, 95))
        if high <= low:
            subset = areas
        else:
            hist, edges = np.histogram(areas, bins=12, range=(low, high))
            peak = int(hist.argmax())
            width = float(edges[1] - edges[0])
            center = 0.5 * float(edges[peak] + edges[peak + 1])
            subset = [value for value in areas if abs(value - center) <= width] or areas
    else:
        subset = areas
    estimate = float(np.median(subset))
    return max(estimate, fallback)


def _split_points(pixels: np.ndarray, count: int, max_points: int) -> List[np.ndarray]:
    """把一个组的像素点分成 count 簇（用于给粘在一起的支架各自画框）。"""
    if count <= 1:
        return [pixels]
    sample = pixels
    if len(pixels) > max_points:
        rng = np.random.default_rng(0)
        sample = pixels[rng.choice(len(pixels), size=max_points, replace=False)]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, _, centers = cv2.kmeans(sample.astype(np.float32), count, None, criteria, 3,
                               cv2.KMEANS_PP_CENTERS)
    distances = ((pixels[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    assignment = distances.argmin(axis=1)
    return [pixels[assignment == k] for k in range(count)]


def objects_from_groups(groups: Sequence[Group], unit_area: float,
                        config: DetectConfig) -> List[StentObject]:
    """按面积把每个组折算成若干个支架对象。

    关键：**先按总面积定总数，再按余数把个数分配到各组**，而不是每组各自四舍五入。
    每组各自四舍五入会累积误差（11.44→11、1.38→1、…，一张图能少掉好几个），
    用"最大余数法"分配后，总数 = round(总材料面积 / A0)，与分组方式无关。
    """
    if not groups or unit_area <= 0:
        return []

    ratios = [group.area / unit_area for group in groups]
    counts = _allocate_counts(ratios, config.min_group_area_ratio)

    objects: List[StentObject] = []
    for group, count, ratio in zip(groups, counts, ratios):
        if count < 1:
            continue                                 # 面积太小，当作残渣
        if count == 1:
            objects.append(StentObject(
                box=group.box, length=group.length, width=group.width, angle=group.angle,
                area=group.area, group_area=group.area, parts=group.parts,
                index_in_group=1, count_in_group=1,
                source="single" if group.parts == 1 else "bridged",
            ))
            continue

        chunks = _split_points(group.pixels, count, config.split_max_points)
        for index, chunk in enumerate(chunks, start=1):
            if len(chunk) < 3:
                continue
            geom = geometry.oriented_geometry(chunk.reshape(-1, 1, 2))
            if geom is None:
                continue
            objects.append(StentObject(
                box=geom.box, length=geom.length, width=geom.width, angle=geom.angle,
                area=int(len(chunk)), group_area=group.area, parts=group.parts,
                index_in_group=index, count_in_group=count, source="split",
            ))
    return objects


def _allocate_counts(ratios: Sequence[float], min_ratio: float) -> List[int]:
    """把总数按面积比例分配成整数个数（最大余数法），总数取 round(Σ比例)。"""
    counts: List[int] = []
    for ratio in ratios:
        count = int(ratio)                      # 向下取整
        if count < 1 and ratio >= min_ratio:
            count = 1                           # 够一个支架的量级，至少算 1 个
        counts.append(count)

    target = int(sum(ratios) + 0.5)
    diff = target - sum(counts)
    remainders = [ratio - int(ratio) for ratio in ratios]

    if diff > 0:
        order = sorted(range(len(ratios)), key=lambda i: remainders[i], reverse=True)
        for step in range(diff):
            counts[order[step % len(order)]] += 1
    elif diff < 0:
        order = sorted(range(len(ratios)), key=lambda i: remainders[i])
        step = 0
        limit = len(order) * 4
        while diff < 0 and step < limit:
            index = order[step % len(order)]
            if counts[index] > 1:
                counts[index] -= 1
                diff += 1
            step += 1
    return counts


def merge_parallel_objects(objects: Sequence[StentObject], image_shape,
                           config: DetectConfig) -> List[StentObject]:
    """目标很少且方向高度一致时，整体并成一个对象（近景单支架样张）。

    只要其中有"从粘连块拆出来的"对象就不合并：拆开说明那块本来就比一个支架大
    （=本来就是多个支架），把它们再合回一个就错了。
    """
    if len(objects) < 2 or not config.single_scene_merge:
        return list(objects)
    if any(item.source == "split" for item in objects):
        return list(objects)
    angles = [item.angle for item in objects]
    if max(angles) - min(angles) > config.group_angle_tolerance:
        return list(objects)
    boxes = [np.asarray(item.box, np.float32).reshape(-1, 2) for item in objects]
    xs = [float(b[:, 0].min()) for b in boxes]
    ys = [float(b[:, 1].min()) for b in boxes]
    xe = [float(b[:, 0].max()) for b in boxes]
    ye = [float(b[:, 1].max()) for b in boxes]
    image_area = float(image_shape[0] * image_shape[1])
    union_area = float((max(xe) - min(xs)) * (max(ye) - min(ys)))
    if image_area <= 0 or union_area / image_area > config.single_scene_area_limit:
        return list(objects)
    merged_points = np.vstack(boxes)
    geom = geometry.oriented_geometry(merged_points.reshape(-1, 1, 2))
    if geom is None:
        return list(objects)
    return [StentObject(
        box=geom.box, length=geom.length, width=geom.width, angle=geom.angle,
        area=int(sum(item.area for item in objects)),
        group_area=int(sum(item.group_area for item in objects)),
        parts=sum(item.parts for item in objects),
        index_in_group=1, count_in_group=1, source="merged",
    )]
