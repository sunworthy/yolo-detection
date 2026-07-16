from ultralytics import YOLO

# 1. 加载一个预训练的轻量级模型（适合在 CPU/普通 GPU 上快速训练）
model = YOLO("yolov8n.pt")

# 2. 开始训练
# data.yaml 配置文件里需要指定你的数据集路径（包含 train 和 val 文件夹）
results = model.train(
    data="./dataset/dataset.yaml",
    epochs=200,  # 训练轮数
    imgsz=640,  # 输入图像大小
    device="cpu",  # 如果有GPU用"0"，没有用"cpu", mac: mps
)
