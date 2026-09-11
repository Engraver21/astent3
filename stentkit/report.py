"""结果落地：标注图、JSON、docx 条件说明。"""

from __future__ import annotations

import json
import zipfile
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import cv2

from . import imageio
from . import pipeline

# 输出文件名一律用 ASCII：既和 V1 的 00_original.png 习惯一致，
# 也避免任何下游工具（压缩包、Excel、Word 批处理）再碰编码问题。
ORIGINAL_NAME = "01_original.png"
ENHANCED_NAME = "02_enhanced.png"
BLACKHAT_NAME = "03_blackhat.png"
MASK_NAME = "04_mask.png"
ANNOTATED_NAME = "05_annotated.png"
IMAGE_NAME = "result.jpg"
JSON_NAME = "result.json"
DOCX_NAME = "conditions.docx"


def new_run_dir(root: Path, prefix: str = "run") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(root) / f"{prefix}_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_image(directory: Path, name: str, image) -> Path:
    path = Path(directory) / name
    if not imageio.imwrite(path, image):
        raise IOError(f"写图失败: {path}")
    return path


def save_json(directory: Path, result: pipeline.DetectionResult, config_dict: dict) -> Path:
    payload = {
        "image": str(result.image_path) if result.image_path else None,
        "count": result.count,
        "candidate_count": result.component_count,
        "threshold": result.threshold,
        "coverage": round(result.coverage, 4),
        "elapsed_sec": round(result.elapsed, 3),
        "scale": {
            "kernel": result.scale.kernel,
            "note": result.scale.note,
            "length": round(result.scale.length, 2),
            "width": round(result.scale.width, 2),
            "sweep": [row.__dict__ for row in result.scale.rows],
        },
        "bounds": {k: round(v, 3) for k, v in result.bounds.items()},
        "threshold_scan": [row.to_dict() for row in result.threshold_rows],
        "objects": [item.to_dict() for item in result.objects],
        "notes": result.notes,
        "config": config_dict,
    }
    path = Path(directory) / JSON_NAME
    # 报告正文保持中文（JSON 是 UTF-8，不受 OpenCV 路径问题影响）
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_docx(directory: Path, lines, name: str = DOCX_NAME) -> Path:
    """不依赖第三方库，直接写一个最小可用的 docx。"""
    paragraphs = "".join(
        f"<w:p><w:r><w:t>{escape(str(line))}</w:t></w:r></w:p>" for line in lines
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{paragraphs}"
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr>'
        "</w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/></Relationships>'
    )
    path = Path(directory) / name
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/document.xml", document)
    return path


def write_run(directory: Path, result: pipeline.DetectionResult, config_dict: dict) -> dict:
    """把一次检测的全部产物写进目录。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "original": save_image(directory, ORIGINAL_NAME, result.image),
        "enhanced": save_image(directory, ENHANCED_NAME, result.gray_eq),
        "blackhat": save_image(directory, BLACKHAT_NAME, result.response),
        "mask": save_image(directory, MASK_NAME, result.mask),
        "annotated": save_image(directory, ANNOTATED_NAME, pipeline.draw_result(result)),
    }
    outputs["image"] = save_image(directory, IMAGE_NAME, pipeline.draw_result(result))
    outputs["json"] = save_json(directory, result, config_dict)
    outputs["docx"] = save_docx(directory, result.summary_lines())
    return outputs
