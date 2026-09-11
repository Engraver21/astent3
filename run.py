#!/usr/bin/env python
"""astent3 入口：一次只检查一张图。

  python run.py detect data/stents4-26.jpg
  python run.py detect data/stents4-26.jpg --out runs/test1 --debug
  python run.py eval
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stentkit import detect_file, load_config           # noqa: E402
from stentkit import evaluate, report                    # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="astent3", description="支架计数 V3")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=None, help="JSON 配置文件（可选）")
    common.add_argument("--no-group", action="store_true", help="关闭「同一支架多段归并」")
    common.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="临时覆盖配置项，例如 --set response_mode=inverse --set length_min_ratio=0.5")

    detect = sub.add_parser("detect", parents=[common], help="检测单张图片")
    detect.add_argument("image", type=Path, help="图片路径")
    detect.add_argument("--out", type=Path, default=None, help="输出目录（默认 runs/run_时间戳）")
    detect.add_argument("--debug", action="store_true", help="打印核扫描与阈值扫描表")
    detect.add_argument("--no-save", action="store_true", help="只打印结果，不写文件")

    evaluate_cmd = sub.add_parser("eval", parents=[common], help="用带真值的样张做回归测试")
    evaluate_cmd.add_argument("--data", type=Path, default=ROOT / "data", help="样张目录")
    evaluate_cmd.add_argument("--csv", type=Path, default=None, help="把结果表存成 CSV")
    evaluate_cmd.add_argument("--debug", action="store_true", help="打印每张图的阈值扫描表")
    return parser


def _overrides(args) -> dict:
    overrides = {"group_enabled": False if args.no_group else None}
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set 需要 KEY=VALUE 形式: {item}")
        key, value = item.split("=", 1)
        try:
            parsed = int(value)
        except ValueError:
            try:
                parsed = float(value)
            except ValueError:
                parsed = {"true": True, "false": False}.get(value.lower(), value)
        overrides[key.strip()] = parsed
    return overrides


def cmd_detect(args) -> int:
    config = load_config(args.config, _overrides(args))
    if not args.image.exists():
        print(f"找不到图片: {args.image}")
        return 2

    result = detect_file(args.image, config)
    print(f"\n=== {args.image.name} ===")
    print(f"图像尺寸: {result.image.shape[1]} x {result.image.shape[0]}")
    print(f"尺度估计: 黑帽核={result.scale.kernel} ({result.scale.note})  "
          f"典型杆长 L={result.scale.length:.1f}px  典型杆宽 W={result.scale.width:.1f}px")
    print(f"尺寸窗口: 长度 {result.bounds['length_min']:.1f}~{result.bounds['length_max']:.1f}px  "
          f"宽度 {result.bounds['width_min']:.1f}~{result.bounds['width_max']:.1f}px  "
          f"面积下限 {result.bounds['area_min']:.0f}px")
    print(f"选用阈值: {result.threshold}   阈值下候选(未归并): {result.component_count}")
    print(f"最终数量: {result.count}   目标覆盖率: {result.coverage:.1%}   用时 {result.elapsed:.2f}s")
    for note in result.notes:
        print(f"提示: {note}")

    if args.debug:
        print("\n[核扫描] 核尺寸 -> 候选数 / 中位杆长 / 中位杆宽")
        for row in result.scale.rows:
            print(f"  k={row.kernel:>4d}  候选={row.count:>5d}  杆长={row.length:7.1f}  杆宽={row.width:6.1f}  "
                  f"Otsu={row.otsu:5.1f}")
        print("\n[阈值扫描] 阈值 -> 尺寸合格候选 / 全部细长连通域 / 稳定度 / 分数")
        for row in result.threshold_rows:
            mark = "  <== 选中" if row.threshold == result.threshold else ""
            print(f"  t={row.threshold:>4d} (x{row.ratio:4.2f})  合格={row.sized_count:>4d}  "
                  f"全部={row.total_count:>5d}  稳定={row.stability:4.2f}  "
                  f"着墨={row.mean_fill:5.1%}  分数={row.score:7.2f}{mark}")
        print("\n[筛除诊断] 选用阈值下每个连通域被哪条规则筛掉")
        print(f"  尺寸窗口: {result.diagnostics['窗口']}")
        print(f"  连通域总数: {result.diagnostics['连通域总数']}  "
              f"筛除统计: {result.diagnostics['筛除统计']}")
        print(f"  通过者中位长度={result.diagnostics['通过者的长度']} "
              f"中位宽度={result.diagnostics['通过者的宽度']}")
        for reason, samples in result.diagnostics["被丢样本"].items():
            if samples:
                print(f"  {reason} 样本 (长, 宽, 面积): {samples}")
    if not args.no_save:
        out_dir = args.out or report.new_run_dir(ROOT / "runs", "detect")
        outputs = report.write_run(out_dir, result, config.to_dict())
        print(f"\n结果目录: {out_dir}")
        for name, path in outputs.items():
            print(f"  {name}: {Path(path).name}")
    return 0


def cmd_eval(args) -> int:
    config = load_config(args.config, _overrides(args))
    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"找不到样张目录: {data_dir}")
        return 2
    rows = evaluate.evaluate_folder(data_dir, config)
    table = evaluate.format_table(rows)
    print(f"\n=== 回归测试: {data_dir} ===")
    print(table)
    if args.debug:
        for path in sorted(data_dir.glob("*.jpg")):
            result = detect_file(path, config)
            print(f"\n--- {path.name} 阈值扫描 ---")
            for row in result.threshold_rows:
                mark = "  <== 选中" if row.threshold == result.threshold else ""
                print(f"  t={row.threshold:>4d} (x{row.ratio:4.2f})  合格={row.sized_count:>4d}  "
                      f"全部={row.total_count:>5d}  稳定={row.stability:4.2f}{mark}")
    if args.csv:
        path = evaluate.save_csv(args.csv, rows)
        print(f"\n已保存: {path}")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # 简写：python run.py 图片路径  ==  python run.py detect 图片路径
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if argv and not argv[0].startswith("-") and argv[0] not in {"detect", "eval"}:
        if Path(argv[0]).suffix.lower() in image_suffixes:
            argv = ["detect"] + argv
        else:
            usage()
            print(f"\n不认识这个参数: {argv[0]}")
            return 2

    # 不带参数时给一份人看的说明，而不是 argparse 的用法报错
    if not argv:
        usage()
        return 0

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "detect":
        return cmd_detect(args)
    if args.command == "eval":
        return cmd_eval(args)
    parser.print_help()
    return 1


def usage() -> None:
    print(
        "支架计数 V3\n"
        "\n"
        "  python run.py detect <图片路径>        检测单张（最常用）\n"
        "  python run.py detect <图片> --debug    额外打印核扫描 / 阈值扫描 / 筛除诊断\n"
        "  python run.py detect <图片> --no-save  只打印结果，不写 runs/ 目录\n"
        "  python run.py eval                     用 data/ 里的带真值样张跑回归\n"
        "  python run.py eval --csv runs/eval.csv 把回归结果存成 CSV\n"
        "  python run.py <图片路径>               上面 detect 的简写\n"
        "  python -m unittest discover -s tests    单元测试\n"
        "\n"
        "例：python run.py detect data/stents4-26.jpg\n"
        "     python run.py data/stents5.jpg --debug\n"
        "\n"
        "在 VS Code 里想按 F5 直接跑，请选左侧「运行和调试」里的\n"
        "「检测：当前打开的图片」或「回归测试：全部样张」（已配好 .vscode/launch.json）。"
    )


if __name__ == "__main__":
    raise SystemExit(main())
