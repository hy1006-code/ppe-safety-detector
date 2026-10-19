# 施工现场安全装备检测

基于 Ultralytics YOLOv8n 的十类目标检测课程项目。

## 已完成结果

累计有效训练100轮，最佳模型位于累计第90轮。验证集 Precision 86.381%，Recall 69.993%，mAP@50 77.311%，mAP@50–95 43.806%。报告内 Kaggle 截图显示 Public Score 0.74007、Private Score 0.70871。

## 环境

Python 3.9.6、PyTorch 2.8.0、Ultralytics 8.4.177。先建立虚拟环境，再安装 requirements-ppe-lock.txt。训练先使用 MPS，反复出现 NaN 后切换 CPU 完成。

## 数据准备

将课堂数据压缩包解压到本仓库根目录，得到 construction_safety_class9_data/ 和 sample_submission.csv。在 construction_safety_class9_data/data.yaml 中，将 path 设置为解压后的数据集目录绝对路径。类别顺序必须保持0–9不变。

## 训练与提交

首次训练：python train_ppe.py --device cpu

从已有第二阶段断点继续：python train_ppe.py --resume runs/ppe/yolov8n_stable/weights/last.pt --completed-epochs 14 --device cpu

首次训练的输出名称为 yolov8n_baseline。以第14轮模型开始低学习率第二阶段：python train_ppe.py --finetune runs/ppe/yolov8n_baseline/weights/best.pt --epochs 86 --completed-epochs 14 --batch 4 --device cpu

指定最佳模型生成提交：python submit.py --checkpoint runs/ppe/yolov8n_stable/weights/best.pt

模型权重、原始数据和训练输出未纳入 Git。本机既有权重位于父目录 runs/ppe/yolov8n_stable/weights/best.pt，需要复制到上述路径后使用。

## 目录说明

train_ppe.py：迁移学习、断点恢复、损失有限值检测、训练状态记录及自动生成提交。
submit.py：测试图片按模板顺序推理，输出归一化坐标与JSON数组。
sample_submission.csv：82张测试图片的提交模板。
requirements-ppe-lock.txt：训练时记录的软件版本。

## 项目边界

本版本未新增网络结构模块；新增工作集中在数据增强配置、稳定训练与输出校验。尚未实现网页摄像头接口。网页实时预测截图需要实际部署并拍摄后补充。本地 Git 仓库没有配置远程地址。

## 许可与数据来源

数据包说明数据来源为 Roboflow Construction Site Safety，许可 CC BY 4.0；Ultralytics 的许可条件参见其官方仓库。
