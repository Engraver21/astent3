"""预处理：灰度、CLAHE、黑帽、Otsu、二值掩膜。"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .config import DetectConfig


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def clahe_image(gray: np.ndarray, config: DetectConfig) -> np.ndarray:
    """限制对比度自适应直方图均衡：压制不均匀光照。"""
    clahe = cv2.createCLAHE(clipLimit=config.clahe_clip, tileGridSize=tuple(config.clahe_grid))
    return clahe.apply(gray)


def blackhat(gray_eq: np.ndarray, kernel_size: int) -> np.ndarray:
    """形态学黑帽：响应"比核窄的暗色结构"（支架是暗的细长目标）。

    注意：**不做 min-max 归一化**。V2 归一化之后几乎每个像素都 >0，
    百分位阈值实际落在噪声上（实测 stents5 的 p70=11，而 Otsu=69）。
    """
    size = int(kernel_size)
    if size % 2 == 0:
        size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(gray_eq, cv2.MORPH_BLACKHAT, kernel)


def otsu_value(values: np.ndarray, *, positive_only: bool = False) -> float:
    """对数组求 Otsu 阈值。"""
    data = values.reshape(-1)
    if positive_only:
        data = data[data > 0]
    if data.size == 0:
        return 0.0
    sample = data.reshape(-1, 1).astype(np.uint8)
    value, _ = cv2.threshold(sample, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(value)


def threshold_mask(response: np.ndarray, threshold: float, close_kernel: int = 3) -> np.ndarray:
    """阈值化 + 轻度闭运算（只补小缺口，不放大目标）。"""
    _, mask = cv2.threshold(response, float(threshold), 255, cv2.THRESH_BINARY)
    if close_kernel and close_kernel >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def connected_components(mask: np.ndarray):
    """连通域统计（含 8 邻接）。"""
    return cv2.connectedComponentsWithStats(mask, connectivity=8)


def dark_ink_mask(gray_eq: np.ndarray, threshold: Optional[float] = None, close_kernel: int = 3) -> np.ndarray:
    """Otsu 反二值化：目标即"比背景暗"的区域（只用于置信度与可视化）。"""
    if threshold is None:
        threshold = otsu_value(gray_eq)
    _, mask = cv2.threshold(gray_eq, float(threshold), 255, cv2.THRESH_BINARY_INV)
    if close_kernel and close_kernel >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask
