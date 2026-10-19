r"""生成本次 Kaggle 目标检测比赛的 submission.csv。

在本目录用 PowerShell 运行：.\.venv-slim\Scripts\python.exe .\submit.py
使用自己的模型：.\.venv-slim\Scripts\python.exe .\submit.py --checkpoint path\to\best.pt
其他电脑需先安装 ultralytics：pip install ultralytics
"""

import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "construction_safety_class9_data"
DEFAULT_CHECKPOINT = ROOT / "runs" / "ppe" / "yolov8n_stable" / "weights" / "best.pt"
DEFAULT_IMAGE_DIR = DATA / "test" / "images"
DEFAULT_SAMPLE = ROOT / "sample_submission.csv"
CLASS_NAMES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="生成施工现场安全目标检测比赛的提交文件")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Ultralytics YOLO 权重路径")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR, help="测试图片目录")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE, help="比赛提供的 sample_submission.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "submission.csv", help="输出 CSV 路径")
    args = parser.parse_args()

    with args.sample.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["image_id", "boxes"]:
            raise ValueError("提交模板必须包含 image_id,boxes 两列")
        image_names = [row["image_id"] for row in reader]
    if len(image_names) != len(set(image_names)):
        raise ValueError("提交模板的 image_id 存在重复")
    missing = [name for name in image_names if not (args.image_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"测试图片缺失：{missing[:5]}")

    model = YOLO(str(args.checkpoint))
    if [model.names[i] for i in range(10)] != CLASS_NAMES:
        raise ValueError("模型类别顺序与比赛的 0–9 类别不一致，请检查权重")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "boxes"])
        for name in image_names:
            result = model.predict(
                source=str(args.image_dir / name), imgsz=640, conf=0.001,
                iou=0.7, max_det=300, verbose=False,
            )[0]
            boxes = [
                [int(category), *coords, confidence]
                for category, coords, confidence in zip(
                    result.boxes.cls.tolist(),
                    result.boxes.xywhn.tolist(),
                    result.boxes.conf.tolist(),
                )
            ]
            writer.writerow([name, json.dumps(boxes, separators=(",", ":"))])
    print(f"saved_submission={args.output.resolve()} count={len(image_names)}")


if __name__ == "__main__":
    main()
