
from ultralytics import YOLO
import os

# 加载训练好的模型
model = YOLO('./tomato/train56/weights/best.pt')

# 图片目录
image_dir = 'D:/YOLOv/datasets/red_tomato_gaosi7/images/val'
# 输出目录
output_dir = 'D:/YOLOv/datasets/red_tomato_gaosi7/images/train56'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 获取图片列表
image_files = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.jpg', '.png'))]

# 进行预测
results = model.predict(source=image_files, save=True, save_txt=True, save_conf=True, project=output_dir)

# 绘制预测结果已经包含在save参数中，预测结果图片会保存在output_dir下的predict目录
# 每张图片的预测文本结果（如果save_txt为True）会保存在对应的子目录下