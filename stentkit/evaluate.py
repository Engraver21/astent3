"""用带真值的样张做回归测试：真值写在文件名末尾，如 stents2-26.jpg。"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from . import pipeline
from .config import DetectConfig

TRUTH_RE = re.compile(r"-(\d+)$")


@dataclass
class EvalRow:
    name: str
    truth: Optional[int]
    predicted: int
    candidates: int
    threshold: int
    kernel: int
    length: float

    @property
    def error(self) -> Optional[float]:
        if self.truth in (None, 0):
            return None
        return (self.predicted - self.truth) / self.truth


def truth_of(path: Path) -> Optional[int]:
    match = TRUTH_RE.search(path.stem)
    return int(match.group(1)) if match else None


def evaluate_folder(data_dir: Path, config: DetectConfig) -> List[EvalRow]:
    rows: List[EvalRow] = []
    for path in sorted(Path(data_dir).glob("*.jpg")):
        result = pipeline.detect_file(path, config)
        rows.append(EvalRow(
            name=path.name, truth=truth_of(path), predicted=result.count,
            candidates=result.component_count, threshold=result.threshold,
            kernel=result.scale.kernel, length=result.scale.length,
        ))
    return rows


def format_table(rows: List[EvalRow]) -> str:
    header = f"{'照片':18s}{'真值':>5s}{'预测':>6s}{'误差':>8s}{'候选':>6s}{'阈值':>6s}{'黑帽核':>7s}{'典型杆长':>9s}"
    lines = [header, "-" * len(header)]
    errors = []
    for row in rows:
        error = row.error
        if error is not None:
            errors.append(abs(error))
        lines.append(
            f"{row.name:18s}"
            f"{('-' if row.truth is None else row.truth):>5}"
            f"{row.predicted:>6d}"
            f"{('  -  ' if error is None else f'{error:+.0%}'):>8s}"
            f"{row.candidates:>6d}"
            f"{row.threshold:>6d}"
            f"{row.kernel:>7d}"
            f"{row.length:>9.1f}"
        )
    if errors:
        lines.append("-" * len(header))
        lines.append(f"有真值样张 {len(errors)} 张，平均绝对误差 {sum(errors)/len(errors):.1%}；"
                     f"最大 {max(errors):.1%}")
    return "\n".join(lines)


def save_csv(path: Path, rows: List[EvalRow]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["照片", "真值", "预测", "误差", "候选数", "阈值", "黑帽核", "典型杆长"])
        for row in rows:
            writer.writerow([
                row.name, "" if row.truth is None else row.truth, row.predicted,
                "" if row.error is None else f"{row.error:+.4f}",
                row.candidates, row.threshold, row.kernel, f"{row.length:.1f}",
            ])
    return path

