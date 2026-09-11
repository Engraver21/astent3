"""开发用：把合成场景/真实图片的内部过程打出来（尺度、阈值扫描、成组、计数）。

用法:
    python tools/inspect_case.py touching
    python tools/inspect_case.py highlight
    python tools/inspect_case.py file data/stents2-26.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (str(ROOT), str(ROOT / "tests")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from stentkit import DetectConfig, detect_image, imageio   # noqa: E402
from stentkit import enhance, grouping, scale as scale_mod  # noqa: E402


def build(kind: str):
    if kind == "touching":
        from test_pipeline import scene_touching_pair
        return scene_touching_pair(), 3
    if kind == "highlight":
        from test_pipeline import scene_highlight_broken
        return scene_highlight_broken(), 2
    raise SystemExit(f"未知场景: {kind}")


def report(image, truth, config: DetectConfig) -> None:
    result = detect_image(image, config)
    print(f"真值={truth}  预测={result.count}")
    print(f"尺度: 核={result.scale.kernel} ({result.scale.note}) L={result.scale.length:.1f} "
          f"W={result.scale.width:.1f}")
    print("核扫描:")
    for row in result.scale.rows:
        print(f"  k={row.kernel:>4d} 候选={row.count:>5d} 杆长={row.length:7.1f} 杆宽={row.width:6.1f} "
              f"Otsu={row.otsu:6.1f}")
    print(f"阈值: {result.threshold}  尺寸合格单块={result.component_count}  "
          f"聚组={result.group_count}  A0={result.unit_area:.0f}")
    print("阈值扫描:")
    for row in result.threshold_rows:
        mark = "  <== 选中" if row.threshold == result.threshold else ""
        print(f"  t={row.threshold:>4d} (x{row.ratio:4.2f}) 合格={row.sized_count:>4d} "
              f"全部={row.total_count:>6d} 稳定={row.stability:4.2f} 着墨={row.mean_fill:6.1%}"
              f" 分数={row.score:8.2f}{mark}")

    # 成组细节
    gray_eq = result.gray_eq
    accept = grouping.material_filter(result.scale, result.unit_area, config)
    groups = grouping.build_groups(result.mask, int(round(config.bridge_ratio * result.scale.length)),
                                   max(12, int(0.08 * result.unit_area)), accept=accept)
    total_area = sum(group.area for group in groups)
    print(f"组细节（组面积 / A0 = 个数）: 共 {len(groups)} 组，"
          f"总材料面积={total_area}，总面积/A0={total_area / max(1.0, result.unit_area):.2f}")
    for group in sorted(groups, key=lambda g: -g.area)[:14]:
        print(f"  面积={group.area:>7d} 块数={group.parts:>3d} 长={group.length:7.1f} "
              f"-> 个数={group.area / max(1.0, result.unit_area):5.2f}")
    print(f"对象来源统计: " + ", ".join(
        f"{src}={sum(1 for o in result.objects if o.source == src)}"
        for src in {o.source for o in result.objects}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="touching / highlight / file")
    parser.add_argument("path", nargs="?", default=None)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    overrides = {}
    for item in args.set:
        key, value = item.split("=", 1)
        overrides[key] = float(value) if value.replace(".", "", 1).isdigit() else value
    config = DetectConfig(**{k: v for k, v in overrides.items() if k in DetectConfig().to_dict()})

    if args.target == "file":
        if not args.path:
            raise SystemExit("file 需要图片路径")
        image = imageio.imread(Path(args.path))
        report(image, "?", config)
    else:
        image, truth = build(args.target)
        report(image, truth, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
