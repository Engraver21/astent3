"""集中管理所有可调参数。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class DetectConfig:
    """检测配置。

    单位说明：所有以 ratio 结尾的参数都是**相对本图尺度**的倍率，不是像素。
    这样同一套配置才能同时吃下"同一堆支架、不同高度/角度"的照片。
    """

    # ---------- 预处理 ----------
    clahe_clip: float = 2.0
    clahe_grid: Tuple[int, int] = (8, 8)
    # 检测用的"响应图"：
    #   blackhat = 形态学黑帽（对"比核窄的暗结构"灵敏，抗不均匀光照，但暗背景照片上噪声大）
    #   inverse  = 255 - CLAHE 灰度（等价于 V1 的反二值化，暗背景照片上更稳）
    # 尺度估计始终走黑帽，与这里无关。
    response_mode: str = "inverse"

    # ---------- 尺度估计 ----------
    # 黑帽核必须比目标"宽"才能把暗色细长目标从背景里挖出来。
    # 扫描后取"候选数随核变化最平缓"的平台中点，避免核太小（噪声爆炸）或太大（目标消失）。
    kernel_sweep: Tuple[int, ...] = (15, 31, 45, 61, 91, 121)
    plateau_tolerance: float = 0.25      # 相邻核之间候选数变化 < 25% 视为同一平台
    scale_aspect_min: float = 2.5        # 参与尺度统计的连通域最小长宽比
    scale_area_ratio: float = 0.12       # 统计尺度时的面积下限 = ratio * 核尺寸^2

    # ---------- 尺寸先验（相对 L / W）----------
    length_min_ratio: float = 0.60       # 合格杆长下限 = 0.60 * L
    length_max_ratio: float = 1.80       # 合格杆长上限 = 1.80 * L（与 V1 的上限口径一致）
    width_min_ratio: float = 0.40        # 合格杆宽下限 = 0.40 * W（滤掉背景细线）
    width_max_ratio: float = 2.50        # 合格杆宽上限 = 2.50 * W
    aspect_min: float = 2.5              # 长宽比下限
    area_min_ratio: float = 0.10         # 面积下限 = ratio * L * W

    # ---------- 阈值选择 ----------
    threshold_ratios: Tuple[float, ...] = (0.60, 0.70, 0.80, 0.90, 1.00, 1.15, 1.30, 1.50, 1.75)
    # 除了"Otsu 的相对倍数"，再补一条绝对扫描线：暗背景照片的 Otsu 未必落在目标对比度上，
    # 只扫 Otsu 附近会漏掉真正合适的那一档（stents1 就是这种情况）。
    absolute_start: int = 20
    absolute_stop: int = 240
    absolute_step: int = 15
    stability_span: int = 2              # 某阈值在 ±n 个相邻阈值里都稳定才算"稳"
    stability_tolerance: float = 0.25    # 相邻阈值候选数变化 < 25% 视为稳定
    min_mean_fill: float = 0.50          # 候选框内着墨率低于此值的阈值直接不参选（挡住噪声阈值）

    # ---------- 形态学 ----------
    close_kernel: int = 3                # 闭运算核（3 表示只补小缺口）

    # ---------- 归并（把同一支架的几段并成一个对象）----------
    # 默认关闭：实测在"多目标"样张上，相邻支架的端部间距常常只有十几像素，
    # 归并会把两个真实支架并成一个（stents2 真值 26 → 归并后 21）。它只适合"画面里只有一个支架"的场景。
    group_enabled: bool = False
    group_gap_ratio: float = 0.15        # 间隙 < 0.15 * L 才考虑归并（太大会把相邻支架并掉）
    group_angle_tolerance: float = 35.0  # 长轴夹角 < 35° 才考虑归并
    # 候选数很少时（画面里本来就没几个目标），把靠近的碎段并成一个对象更合理：
    # 单支架样张上，一个支架常被分成 2~5 段，不并就会把 1 个数成 5 个。
    auto_group_threshold: int = 6
    single_scene_gap_ratio: float = 0.25
    # 目标很少 + 方向高度一致 → 判为"同一个支架被拆成几段"，整体并成一个对象。
    # 默认关闭：多根支架并排摆放时（方向也一致），这条规则会把它们合成 1 个。
    # 只有"整张图就是一支支架的近景"这种用法才建议打开（--set single_scene_merge=true）。
    single_scene_merge: bool = False
    single_scene_area_limit: float = 0.35   # 合并后的外接框面积不得超过整图的这个比例

    # ---------- 成组计数（解决"贴得近"与"高光打断"）----------
    # 关键思路：一个支架被高光打断成几段、或者两个支架贴在一起粘成一块，
    # 单看连通域都会数错；所以先按距离把碎块聚成"组"，再用"一个支架的典型面积 A0"
    # 去算这一组里到底有几个：n = round(组面积 / A0)。
    # 组并得大一点也没关系，因为计数靠面积不靠块的个数。
    bridge_ratio: float = 0.20            # 聚组半径 = 0.20 * L（用于把同一支架的碎段并到一起）
    min_group_area_ratio: float = 0.45    # 组面积 < 0.45 * A0 视为残渣/噪点，不计
    unit_area_min_ratio: float = 0.30     # A0 兜底下限：0.30 * L * W
    # A0 的取法：trimmed=先按 25 分位定"单块量级"，剔除明显更大的（疑似粘连块）后取中位数；
    #            median=直接取中位数；lower_half=下半部分中位数；p25=25 分位。
    # 偏小会把大组多算 1 个，偏大会把粘连的两个少算成 1 个；四者用真值样张实测后选 trimmed。
    unit_area_mode: str = "trimmed"
    unit_area_trim_factor: float = 1.45   # trimmed 模式下"多大算疑似粘连块"的倍率（相对 25 分位）
    split_max_points: int = 4000          # 一次 k-means 最多用多少像素点（大图抽样，保速度）
    # "计数材料"的形状门槛：长宽比/杆宽按尺度严格卡住（把噪声块挡掉），
    # 只放宽长度——因为粘连会让它变长、高光会让它变短。
    fragment_area_ratio: float = 0.20     # 材料面积下限 = 0.20 * A0
    fragment_length_min_ratio: float = 0.20
    fragment_length_max_ratio: float = 3.00

    # ---------- 输出 ----------
    numbering: bool = True                # 在结果图上给每个识别到的支架标序号
    number_font_scale: float = 0.0        # 0 = 按图幅自动；也可手工指定

    # ---------- 兜底（默认关闭）----------
    # V2 的 Hough 兜底在纹理背景上会撒出成百上千个框（1 个支架 → 46 个框），
    # 因此默认关闭；确需启用时还必须满足填充率与数量上限等门控。
    use_hough_fallback: bool = False
    hough_max_boxes: int = 8
    hough_min_fill: float = 0.55

    # ---------- 其他 ----------
    max_components: int = 5000           # 安全阀：连通域异常多时直接判定尺度异常
    verbose: bool = True
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["kernel_sweep"] = list(self.kernel_sweep)
        data["threshold_ratios"] = list(self.threshold_ratios)
        data["clahe_grid"] = list(self.clahe_grid)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "DetectConfig":
        known = {f for f in cls.__dataclass_fields__}
        payload = {k: v for k, v in data.items() if k in known}
        for key in ("kernel_sweep", "threshold_ratios", "clahe_grid"):
            if payload.get(key) is not None:
                payload[key] = tuple(payload[key])
        return cls(**payload)


def load_config(path: Optional[Path] = None, overrides: Optional[dict] = None) -> DetectConfig:
    """从 JSON 文件加载配置，并叠加命令行覆盖项。"""
    config = DetectConfig()
    if path is not None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        config = DetectConfig.from_dict(payload)
    if overrides:
        payload = config.to_dict()
        payload.update({k: v for k, v in overrides.items() if v is not None})
        config = DetectConfig.from_dict(payload)
    return config
