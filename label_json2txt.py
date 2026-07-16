import os
import json
import shutil
import math
from glob import glob

script_dir = os.path.dirname(os.path.abspath(__file__))
print(script_dir)
# --- 路径与配置 ---
src_dir = os.path.abspath(os.path.join(script_dir, "dataset_labelme"))
output_dir = os.path.abspath(os.path.join(script_dir, "dataset"))

# 1. 修正映射：把你的 "circle1" 映射为 YOLO 的第 0 类
class_mapping = {"circle1": 0}

# 创建 YOLO 目录结构
for folder in ["images/train", "images/val", "labels/train", "labels/val"]:
    os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

json_files = glob(os.path.join(src_dir, "*.json"))

if not json_files:
    print(f"错误：在 {src_dir} 找不到任何 .json 文件！")

for json_path in json_files:
    base_name = os.path.splitext(os.path.basename(json_path))[0]

    # 兼容各种图片格式
    img_path = None
    for ext in [".png", ".jpg", ".jpeg", ".JPG", ".PNG"]:
        potential_img = os.path.join(src_dir, base_name + ext)
        if os.path.exists(potential_img):
            img_path = potential_img
            break

    if img_path is None:
        print(f"找不到 {base_name} 对应的图片，跳过。")
        continue

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_h = data["imageHeight"]
    img_w = data["imageWidth"]
    yolo_lines = []

    for shape in data["shapes"]:
        label = shape["label"]
        if label not in class_mapping:
            print(f"警告：忽略了未识别的标签 '{label}'")
            continue
        class_id = class_mapping[label]

        points = shape["points"]

        # === 核心算法：针对 Labelme 圆形标注的解析 ===
        if shape.get("shape_type") == "circle":
            # 第一个点是圆心
            cx, cy = points[0][0], points[0][1]
            # 第二个点是圆周上的点
            px, py = points[1][0], points[1][1]

            # 利用两 point 间距离公式求半径 R
            r = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)

            # YOLO 矩形边界框的绝对中心点就是圆心
            center_x = cx
            center_y = cy
            # 矩形边界框的绝对宽度和高度就是直径 (2 * R)
            box_w = 2 * r
            box_h = 2 * r
        else:
            # 如果你混用了矩形或多边形标注，走标准逻辑
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            box_w = xmax - xmin
            box_h = ymax - ymin
            center_x = xmin + box_w / 2.0
            center_y = ymin + box_h / 2.0

        # 归一化到 0~1 之间
        x = center_x / img_w
        y = center_y / img_h
        w = box_w / img_w
        h = box_h / img_h

        yolo_lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    if yolo_lines:
        # 强行把唯一的图片和标签同时写入训练集和验证集进行过拟合通路测试
        for split in ["train", "val"]:
            shutil.copy(
                img_path,
                os.path.join(output_dir, "images", split, os.path.basename(img_path)),
            )

            txt_output_path = os.path.join(
                output_dir, "labels", split, base_name + ".txt"
            )
            with open(txt_output_path, "w") as f_out:
                f_out.write("\n".join(yolo_lines))
        print(f"🎉 成功！{base_name}.json 中的标注已转换为 YOLO 格式！")
