import cv2
import numpy as np

img = cv2.imread("dataset_labelme/chopsticks2.png")
if img is None:
    raise Exception("img not found")
result_img = img.copy()
# 灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 2. 图像预处理：去除金属表面的细小噪声
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# # 边缘检测
edges = cv2.Canny(blurred, 50, 150)
# # 形态学闭运算-连接断裂边缘
kernel = np.ones((3, 3), np.uint8)
edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

# cv2.imshow("edges_closed", edges_closed)
# 开运算, 去除小噪点
# edges_clean = cv2.morphologyEx(edges_closed, cv2.MORPH_OPEN, kernel)
#
circles = cv2.HoughCircles(
    edges_closed,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=40,
    param1=90,
    param2=30,
    minRadius=23,  # 最小半径
    maxRadius=50,  # 最大半径
)

# 4. 如果检测到圆，则在原图上绘制
count = 0
if circles is not None:
    # 将浮点型坐标和半径四舍五入并转换为整数
    circles = np.uint16(np.around(circles))
    print(circles)
    for i in circles[0, :]:
        x, y, r = i[0], i[1], i[2]

        # 绘制圆形轮廓 (绿色, 线宽为3)
        cv2.circle(result_img, (x, y), r, (0, 255, 0), 3)

        # 绘制圆心 (红色实心小圆, 半径为2)
        cv2.circle(result_img, (x, y), 2, (0, 0, 255), -1)  # 使用 -1 填充圆心
        count += 1
print("count: ", count)
cv2.imshow("img", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()