# 施工现场安全装备检测

基于 Ultralytics YOLOv8n 的十类目标检测课程项目。

## 已完成结果

累计有效训练100轮，最佳模型位于累计第90轮。验证集 Precision 86.381%，Recall 69.993%，mAP@50 77.311%，mAP@50–95 43.806%。报告内 Kaggle 截图显示 Public Score 0.74007、Private Score 0.70871。

## 演示视频

[demo/ppe_detection_demo.mp4](demo/ppe_detection_demo.mp4)：检测效果录屏（由原始 170MB 录屏压缩至约 2MB，H.264），在 GitHub 仓库页面可直接在线播放。

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

本版本未新增网络结构模块；新增工作集中在数据增强配置、稳定训练与输出校验。本地 Git 仓库没有配置远程地址。

## 网页摄像头实时检测

在这台 Mac 上双击“运行摄像头.command”，或者使用项目虚拟环境执行 python camera_server.py。浏览器打开 http://127.0.0.1:8765 。

将最佳权重放在 models/best.pt；在本机也可自动读取父目录 runs/ppe/yolov8n_stable/weights/best.pt。默认使用第90轮最佳权重及CPU推理，默认置信度0.25，检测输入尺寸640，NMS IoU 0.7，每帧最多显示100个框。可以使用 --checkpoint 指定其他权重、--port 指定端口。

点击“打开摄像头并检测”，允许 Chrome 或 Safari 使用摄像头。若 macOS 阻止访问，请在“系统设置 → 隐私与安全性 → 摄像头”允许当前浏览器。首次使用可点击“先用示例图片检查模型”验证连接；页面会明确标注示例图片不属于摄像头实拍。

戴好安全帽与安全背心，分别在正面、侧面或改变距离、不同光照下拍摄，点击“保存检测截图”。截图包含检测框、中文类别、置信度、模型名称和时间。也可用 macOS 的 Shift+Command+4 截取整个网页作为报告实测证据。点击“停止检测”会释放摄像头；终端按 Ctrl+C 结束本地服务。

摄像头图像只发送到本机127.0.0.1服务，服务器不自动保存画面。网页每次只发送一帧，返回后再发送下一帧，避免堆积请求；显示的框对应同一帧图像。实际帧率取决于本机负载，页面显示真实处理耗时与更新速度。

网页文件位于 web/，服务入口为 camera_server.py。浏览器使用 getUserMedia 获取图像，通过 JPEG 请求调用本机 YOLOv8n，前端用 Canvas 绘制结果。无需 Flask、FastAPI 或额外前端依赖。

## 许可与数据来源

数据包说明数据来源为 Roboflow Construction Site Safety，许可 CC BY 4.0；Ultralytics 的许可条件参见其官方仓库。
