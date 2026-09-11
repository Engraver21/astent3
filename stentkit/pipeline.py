"""主流程编排：一张图进，一个结果出。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

from . import candidates as cand_mod
from . import diagnose as diagnose_mod
from . import enhance
from . import geometry
from . import grouping
from . import imageio
from . import scale as scale_mod
from . import selector
from .config import DetectConfig


@dataclass
class DetectionResult:
    image_path: Optional[Path]
    image: np.ndarray
    gray_eq: np.ndarray
    response: np.ndarray
    mask: np.ndarray
    ink_mask: np.ndarray
    scale: scale_mod.ScaleEstimate
    bounds: dict
    threshold: int
    threshold_rows: List[selector.ThresholdRow]
    candidates: List[cand_mod.Candidate]
    objects: List[grouping.StentObject]
    unit_area: float = 0.0
    group_count: int = 0
    coverage: float = 0.0
    elapsed: float = 0.0
    notes: List[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    config_snapshot: dict = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.objects)

    @property
    def component_count(self) -> int:
        return len(self.candidates)

    def summary_lines(self) -> List[str]:
        scale = self.scale
        sources = {}
        for item in self.objects:
            sources[item.source] = sources.get(item.source, 0) + 1
        source_text = "，".join(
            f"{name}={count}" for name, count in sorted(sources.items())
        ) or "无"
        lines = [
            "支架检测 V3 条件说明",
            f"输入图像: {self.image_path.name if self.image_path else '(内存图像)'}",
            f"图像尺寸: {self.image.shape[1]} x {self.image.shape[0]}",
            f"最终检测数量: {self.count}",
            f"序号范围: 1 ~ {self.count}（结果图上每个支架都带序号）",
            f"尺寸合格的单块数: {self.component_count}",
            f"聚组数: {self.group_count}",
            f"单支架典型面积 A0: {self.unit_area:.0f} px（组面积 / A0 四舍五入得到个数）",
            f"对象来源: {source_text}（single=单块，bridged=高光/遮挡打断后拼回，split=粘连块拆开）",
            f"选用阈值: {self.threshold}",
            f"尺度估计: 黑帽核={scale.kernel} ({scale.note})，典型杆长 L={scale.length:.1f} px，"
            f"典型杆宽 W={scale.width:.1f} px",
            f"长度窗口: {self.bounds['length_min']:.1f} ~ {self.bounds['length_max']:.1f} px",
            f"宽度窗口: {self.bounds['width_min']:.1f} ~ {self.bounds['width_max']:.1f} px",
            f"面积下限: {self.bounds['area_min']:.0f} px",
            f"掩膜像素覆盖率: {self.coverage:.1%}",
            f"响应图: {self.config_snapshot.get('response_mode', 'inverse')}"
            f"    线段兜底: {'启用' if self.config_snapshot.get('use_hough_fallback') else '未启用'}",
            "说明: 阈值选择按「与尺度先验一致 + 跨阈值稳定」打分，不取候选数最多的一档；"
            "跨阈值支持度仅作为置信度记录，不参与删除。",
        ]
        lines.extend(self.notes)
        return lines


def detect_image(image: np.ndarray, config: Optional[DetectConfig] = None,
                 image_path: Optional[Path] = None) -> DetectionResult:
    started = time.time()
    config = config or DetectConfig()
    image = image.copy()

    gray = enhance.to_gray(image)
    gray_eq = enhance.clahe_image(gray, config)
    scale = scale_mod.estimate_scale(gray_eq, config)
    bounds = scale_mod.size_window(scale, config)
    if config.response_mode == "inverse":
        response = cv2.bitwise_not(gray_eq)
    else:
        response = enhance.blackhat(gray_eq, scale.kernel)
    ink_mask = enhance.dark_ink_mask(gray_eq, close_kernel=config.close_kernel)

    threshold, rows, chosen, per_threshold = selector.evaluate_thresholds(
        response, scale, ink_mask, config
    )
    selector.compute_support(per_threshold, chosen)

    mask = enhance.threshold_mask(response, threshold, config.close_kernel)
    candidates = list(chosen)                 # 尺寸合格的单块，用于估计"一个支架的典型面积"
    notes: List[str] = []

    # 成组计数：贴在一起的两个支架（一块太大）与中间被高光打断的支架（几块太小）
    # 都由"组面积 / 单个支架典型面积"这一步统一处理。
    unit_area = grouping.estimate_unit_area([item.area for item in candidates], scale, config)
    accept = grouping.material_filter(scale, unit_area, config)
    groups = grouping.build_groups(mask, int(round(config.bridge_ratio * scale.length)),
                                   min_area=max(12, int(0.08 * unit_area)), accept=accept)
    objects = grouping.objects_from_groups(groups, unit_area, config)
    if 0 < len(objects) <= config.auto_group_threshold:
        objects = grouping.merge_parallel_objects(objects, image.shape, config)
        if len(objects) == 1 and groups:
            notes.append("候选很少且方向一致，已判为「同一个支架被拆成几段」，合并为 1 个对象。")
    if config.use_hough_fallback:
        for item in cand_mod.hough_candidates(mask, scale, config):
            objects.append(grouping.StentObject(
                box=item.box, length=item.length, width=item.width, angle=item.angle,
                area=item.area, group_area=item.area, parts=1,
                index_in_group=1, count_in_group=1, source="hough",
            ))
    split_count = sum(1 for item in objects if item.source == "split")
    bridged_count = sum(1 for item in objects if item.source == "bridged")
    if split_count:
        notes.append(f"有 {split_count} 个支架是从粘连块里拆出来的（贴得近的情况）。")
    if bridged_count:
        notes.append(f"有 {bridged_count} 个支架是由多个碎块拼回来的（高光/遮挡打断的情况）。")

    total_components = int(enhance.connected_components(mask)[0] - 1)
    if total_components > config.max_components:
        notes.append(f"警告: 阈值 {threshold} 下连通域数 {total_components} 异常多，"
                     f"常见于背景纹理/光照极不均匀，建议检查原图或提高黑帽核。")

    covered = np.zeros(mask.shape[:2], np.uint8)
    for item in objects:
        cv2.fillPoly(covered, [np.int32(item.box)], 255)
    # 覆盖率按"检测掩膜"算：用 Otsu 反二值图当分母会在暗背景照片上失真
    total_target = int(cv2.countNonZero(mask))
    coverage = (cv2.countNonZero(cv2.bitwise_and(mask, covered)) / total_target) if total_target else 0.0

    return DetectionResult(
        image_path=image_path, image=image, gray_eq=gray_eq, response=response,
        mask=mask, ink_mask=ink_mask, scale=scale, bounds=bounds,
        threshold=threshold, threshold_rows=rows, candidates=candidates,
        objects=objects, coverage=float(coverage),
        unit_area=float(unit_area), group_count=len(groups),
        elapsed=time.time() - started, notes=notes,
        diagnostics=diagnose_mod.explain_components(mask, scale, config, window=bounds),
        config_snapshot=config.to_dict(),
    )


def detect_file(path: Path, config: Optional[DetectConfig] = None) -> DetectionResult:
    image = imageio.imread(path)
    if image is None:
        raise FileNotFoundError(f"读不到图像: {path}")
    return detect_image(image, config, image_path=Path(path))


def draw_result(result: DetectionResult, thickness: int = 2) -> np.ndarray:
    """把每个识别到的支架画出来并标序号。

    颜色区分来源：绿色=单块直接识别；橙色=高光/遮挡打断后拼回；蓝色=从粘连块里拆开。
    """
    canvas = result.image.copy()
    source_color = {
        "single": (0, 200, 0),
        "bridged": (0, 165, 255),
        "split": (255, 120, 0),
        "merged": (200, 0, 200),
        "hough": (128, 128, 128),
    }
    font_scale = result.config_snapshot.get("number_font_scale") or 0.0
    if not font_scale:
        font_scale = max(0.6, min(canvas.shape[:2]) / 900.0)
    font_scale = float(font_scale)
    badge_thickness = max(1, int(round(font_scale * 2)))

    for index, item in enumerate(result.objects, start=1):
        color = source_color.get(item.source, (0, 200, 0))
        cv2.polylines(canvas, [np.int32(item.box)], True, color, thickness)
        x, y = item.box[0]
        _draw_number_badge(canvas, index, int(x), int(y), color, font_scale, badge_thickness)
    return canvas


def _draw_number_badge(canvas: np.ndarray, number: int, x: int, y: int,
                       color, font_scale: float, thickness: int) -> None:
    """在框角画一个实心序号牌，保证在任何底色上都看得清。"""
    text = str(number)
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                                 font_scale, thickness)
    pad = max(4, int(round(font_scale * 6)))
    box_w, box_h = text_w + pad * 2, text_h + baseline + pad
    height, width = canvas.shape[:2]
    x0 = min(max(0, x), max(0, width - box_w))
    y0 = min(max(0, y - box_h), max(0, height - box_h))
    overlay = canvas[y0:y0 + box_h, x0:x0 + box_w]
    cv2.rectangle(overlay, (0, 0), (box_w - 1, box_h - 1), color, -1)
    cv2.putText(canvas, text, (x0 + pad, y0 + box_h - pad - baseline + text_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
