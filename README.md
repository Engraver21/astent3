# astent3 — 支架计数（V3）

V1（astent/first.py）和 V2（astent2/first_v2.py）的问题诊断与真值对照见 docs/findings.md。
V3 是在此基础上的重写，原则是：先稳、再准、最后才谈花哨。

## V3 相对 V1/V2 的关键改动

| 问题（V1/V2） | V3 的做法 |
|---|---|
| V2 的 hough_fallback 在 OpenCV 5 下崩溃（lines[:, 0]） | 删除该支路。带严格门控的线段兜底放在 candidates.py:hough_candidates，默认关闭 |
| V2 只要候选 < 17 个就在全图撒 Hough 线（1 个支架的照片给出 46 个框） | 不再用"候选少"当触发条件，默认完全不启用 |
| 两版都用"候选数最多的阈值"当选最佳，等于奖励碎片化 | selector.py 用"与尺度先验一致的候选数"打分，并要求阈值稳定（平台），不再单纯取最大 |
| V2 把 V1 唯一的自适应机制（按图中位杆长定长度窗口）删掉，改成固定像素 | scale.py 先用黑帽核尺寸扫描找平台，由平台处的连通域统计出本图典型杆长 L、杆宽 W，之后所有阈值/面积/长度窗口都按 L、W 缩放 |
| CLAHE 反二值化对光照敏感（V1）；blackhat 之后再做 NORM_MINMAX 把阈值变成噪声分位（V2） | 用原始黑帽响应 + Otsu 定基准阈值，只在 Otsu 附近扫描；不做 min-max 归一化 |
| 跨阈值支持度被当硬过滤器，把只在窄阈值区间可见的目标删掉（stents1 真值 20，V1 只给 15） | 支持度只作为置信度字段保留在结果里，不参与删除 |

## 目录结构

```
astent3/
├── run.py                  # 唯一入口：detect / eval
├── stentkit/
│   ├── config.py           # 所有参数集中在这里（含注释与默认值）
│   ├── geometry.py         # 最小外接矩形、IoU、多边形间距等纯几何
│   ├── enhance.py          # 灰度 / CLAHE / 黑帽 / Otsu / 掩膜
│   ├── scale.py            # 尺度估计（核平台 + 典型杆长/杆宽）
│   ├── candidates.py       # 候选提取、归并、（可选）线段兜底
│   ├── selector.py         # 阈值选择与评分
│   ├── pipeline.py         # 主流程编排
│   ├── report.py           # 标注图 / JSON / docx 报告
│   └── evaluate.py         # 和文件名真值对比，输出精度表
├── data/                   # 已标真值的样张（文件名后缀 -N 为真实数量）
├── docs/findings.md        # V1/V2 诊断结论、真值对照表
├── tests/                  # unittest（不需要第三方测试库）
└── runs/                   # 每次运行的结果目录
```

## 用法

```bash
# 检查一张（每次只处理一张，符合现场用法）
python run.py detect data/stents4-26.jpg

# 指定输出目录
python run.py detect data/stents4-26.jpg --out runs/test1

# 用带真值的样张做回归测试（打印漏检/误检表）
python run.py eval
python run.py eval --data data --csv runs/eval.csv

# 单元测试
python -m unittest discover -s tests
```

输出目录内容：01_original.png、02_enhanced.png、03_blackhat.png、04_mask.png、05_annotated.png、
result.jpg、conditions.docx、result.json（含全部参数、阈值扫描表、每个框的坐标与置信度）。

> 输出文件名一律用 ASCII。原因是 Windows 上 `cv2.imwrite` 会按本地 ANSI 代码页解释路径，
> 中文名会落盘成乱码（`01_原图.png` → `01_鍘熷浘.png`）；读写统一走 `stentkit/imageio.py`
> （Python 开文件 + `cv2.imencode/decode`），所以中文名的**输入**图片也能正常读。
> 报告正文（result.json / conditions.docx）仍是中文。

## 真值样张命名

文件名末尾 -N 表示该图里支架的真实数量，例如 stents2-26.jpg 表示 26 个。
eval 子命令靠这个后缀自动对比；没有后缀的图会被跳过（当前 stents5/stents6 还没有真值）。

## 当前精度

```
python run.py eval
stents1-20.jpg     真值 20  预测 17   -15%
stents2-26.jpg     真值 26  预测 26    +0%
stents3-26.jpg     真值 26  预测 26    +0%
stents4-26.jpg     真值 26  预测 26    +0%
stents7-57.jpg     真值 57  预测 53    -7%
```

五张有真值样张的平均绝对误差 5.2%，最大 15%。同场景不同拍的两张（stents5/stents6）
给出 128 / 125，相差约 2%（改动前是 10%），可以用作"换角度/换光线下稳定性"的自检指标。
详细诊断与证据见 docs/findings.md。

误差方向并不一致（-15% / +4% / 0 / 0 / -7%），说明还有约 5% 的随机误差，
要继续压下去就得再加几张带真值的样张（改名成 `名字-数量.jpg` 就自动进回归表）。

## 计数模型（V3 与 V1/V2 最大的不同）

单看"连通域"会同时犯两个错，而它们方向相反：

* **贴得近**：两个支架粘成一块，长度超过上限 → 整块被丢掉（漏检）；
* **有高光**：一个支架被亮斑打断成两段 → 每段都太短 → 两段都被丢掉（漏检）。

所以 V3 不再直接数连通域，而是两步：

1. **聚组**：按 `0.2 × L` 的半径把碎块聚成组（同一支架的碎段会被并到一组）；
2. **按面积归一化计数**：`个数 = round(总材料面积 / A0)`，A0 是本图"一个支架的典型着墨面积"。
   总数先由总面积定下来，再按余数（最大余数法）分配到各组，避免每组各自四舍五入累积误差
   （每组各自取整时，11.44→11、1.38→1… 一张图能少掉好几个）。

组并得大一点不影响结果，因为个数由面积决定、不由块数决定：两个支架粘在一起，面积就是 2 倍；
一个支架被打断成两段，两段加起来仍然只是 1 倍。
A0 用"先按 25 分位定量级、剔除疑似粘连块后取中位数"的稳健估计（`unit_area_mode=trimmed`），
避免粘连块把 A0 抬高导致少算。

## 结果图上的颜色与序号

每个识别到的支架都有一个白字实心序号牌（1..N），框的颜色表示它的来源：

| 颜色 | 含义 |
|---|---|
| 绿色 | 单块直接识别（最普通的情况） |
| 橙色 | 高光/遮挡打断后拼回来的（问题 2） |
| 蓝色 | 从粘连块里拆出来的（问题 1） |
| 紫色 | 单支近景整体合并后的对象 |

## 单支近景（可选）

现场不会有这种场景（用户已确认），所以 `stents-1.jpg` 已移到
`data/ignored_single_closeup/`，不参与回归。功能本身留着，万一要拍单支近景可以打开：

```bash
python run.py detect data/ignored_single_closeup/stents-1.jpg --set single_scene_merge=true
```

默认关闭是因为反向的坑更常见：多根支架并排摆放时方向也一致，打开它会把整排并成 1 个。

调参工具：`python tools/param_sweep.py`（用带真值的样张扫描参数网格，按平均绝对误差排序）。
临时改参数：`python run.py detect data/stents1-20.jpg --set response_mode=blackhat --set length_min_ratio=0.5`
