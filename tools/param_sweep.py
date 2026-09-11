"""参数扫描：用带真值的样张挑默认参数。

用法:
    python tools/param_sweep.py                 # 跑内置网格
    python tools/param_sweep.py --top 12        # 只看最好的若干组
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stentkit import DetectConfig, detect_file          # noqa: E402
from stentkit.evaluate import truth_of                   # noqa: E402


def score_config(data_dir: Path, config: DetectConfig) -> tuple:
    errors = []
    details = []
    for path in sorted(data_dir.glob("*.jpg")):
        truth = truth_of(path)
        if truth is None:
            continue
        predicted = detect_file(path, config).count
        error = abs(predicted - truth) / truth
        errors.append(error)
        details.append(f"{path.stem}:{predicted}/{truth}")
    if not errors:
        return float("inf"), ""
    return sum(errors) / len(errors), " ".join(details)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    grid = list(itertools.product(
        ["blackhat", "inverse"],        # response_mode
        [0.50, 0.60],                   # length_min_ratio
        [1.60, 1.80, 2.00],             # length_max_ratio
        [0.25, 0.40],                   # width_min_ratio
        [0.05, 0.10],                   # area_min_ratio
    ))
    results = []
    for mode, lmin, lmax, wmin, amin in grid:
        config = DetectConfig(
            response_mode=mode,
            length_min_ratio=lmin,
            length_max_ratio=lmax,
            width_min_ratio=wmin,
            area_min_ratio=amin,
            verbose=False,
        )
        mean_error, details = score_config(args.data, config)
        results.append((mean_error, mode, lmin, lmax, wmin, amin, details))

    results.sort(key=lambda row: row[0])
    print(f"{'平均绝对误差':>12s}  {'响应':>8s} {'长下限':>6s} {'长上限':>6s} "
          f"{'宽下限':>6s} {'面积':>5s}  明细")
    for mean_error, mode, lmin, lmax, wmin, amin, details in results[:args.top]:
        print(f"{mean_error:12.1%}  {mode:>8s} {lmin:6.2f} {lmax:6.2f} {wmin:6.2f} {amin:5.2f}  {details}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

