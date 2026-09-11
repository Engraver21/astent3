"""图片读写封装（正确处理中文/非 ASCII 路径）。

Windows 上 `cv2.imread` / `cv2.imwrite` 会把路径按本地 ANSI 代码页解释，
遇到中文名会出现两种问题：
  1. 文件名落盘变乱码（"01_原图.png" 变成 "01_鍘熷浘.png"）；
  2. 直接读写失败（返回 None / False），且不报错。

这里统一改成"用 Python 打开文件 → 交给 cv2.imdecode / cv2.imencode"，
路径交给 Python 处理，编码交给 OpenCV 处理，两边都不越界。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

PathLike = Union[str, Path]


def imread(path: PathLike, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """读图，路径支持中文。读不到返回 None。"""
    file_path = Path(path)
    if not file_path.is_file():
        return None
    try:
        buffer = np.frombuffer(file_path.read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, flags)


def imwrite(path: PathLike, image: np.ndarray, params: Optional[list] = None) -> bool:
    """写图，路径支持中文。返回是否成功（OpenCV 原生 imwrite 失败时是静默的）。"""
    file_path = Path(path)
    suffix = file_path.suffix.lower() or ".png"
    ok, buffer = cv2.imencode(suffix, image, params or [])
    if not ok:
        return False
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(buffer.tobytes())
    return True

