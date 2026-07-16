import cv2
from ultralytics import YOLO


def adaptive(_count):
    # 对局部进行二值化，把整个筷子头的截面轮廓真正抠出来, 用局部自适应阈值（高斯法）替代大津法，专门抓取阴影边缘
    roi_thresh = cv2.adaptiveThreshold(
        roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2
    )

    # 计算这个局部实体的真实物理质心
    moments = cv2.moments(roi_thresh)
    if moments["m00"] != 0:
        # 算出来的质心是相对于局部 ROI 的，要加上左上角坐标 (x1, y1) 还原回原图
        center_x = int(moments["m10"] / moments["m00"]) + x1
        center_y = int(moments["m01"] / moments["m00"]) + y1
    else:
        # 如果局部二值化失败，退回到矩形中心作为保底
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

    # === 半径修正：让圆圈稍微扩大一点，完美包裹筷子头 ===
    width = x2 - x1
    height = y2 - y1
    # 之前是除以 4，圆圈太小；我们乘上一个 0.6 的系数，手动让圆圈张开
    radius = int(max(width, height) * 0.5)

    cv2.circle(result_img, (center_x, center_y), radius, (0, 255, 0), 2)
    cv2.circle(result_img, (center_x, center_y), 3, (0, 0, 255), -1)

    cv2.putText(
        result_img,
        f"{_count + 1}",
        (center_x - 5, center_y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
    )


# 1. 加载你自己训练好的最佳模型
model = YOLO("runs/detect/train/weights/best.pt")

# 2. 读取测试图片
img_path = "dataset_labelme/chopsticks2.png"
img = cv2.imread(img_path)
if img is None:
    raise Exception(f"图片未找到，请检查路径是否正确: {img_path}")
result_img = img.copy()

# 3.把原图转成单通道灰度图，供后面局部修正使用
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 4. 模型推理
results = model.predict(source=img, conf=0.25)

# 5. 解析预测结果
count = 0
for result in results:
    boxes = result.boxes
    for box in boxes:
        # 1. 获取 YOLO 预测的原始矩形坐标
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # 限制边界，防止裁剪越界
        h_img, w_img = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        # 提取这根筷子头所在的小局部区域
        roi = gray[y1:y2, x1:x2]

        adaptive(count)
        count += 1

print(f"YOLO 检测并转换为圆形的筷子总数: {count}")

# 6. 显示最终结果
cv2.imshow("YOLO detect result", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
