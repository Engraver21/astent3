"""支架计数工具包（V3）。

设计原则：
  1. 先把尺度估计出来（自适应），再用尺度约束一切阈值与几何筛选；
  2. 阈值选择不奖励"候选多"，只奖励"与尺度先验一致且稳定"；
  3. 任何会大幅增加候选数的兜底逻辑都必须默认关闭、且有硬上限。
"""

from .config import DetectConfig, load_config
from .pipeline import DetectionResult, detect_file, detect_image

__all__ = [
    "DetectConfig",
    "load_config",
    "DetectionResult",
    "detect_file",
    "detect_image",
]

