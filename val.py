from ultralytics import YOLO
import os
from multiprocessing import freeze_support


def main():
    # 加载训练好的模型
    model = YOLO('./tomato/train58/weights/best.pt')

    # 假设 data.yaml 文件的路径
    data_yaml_path = 'D:/YOLOv/datasets/red_tomato_gaosi7/data.yaml'

    # 检查 data.yaml 文件是否存在
    if not os.path.exists(data_yaml_path):
        print(f"Data.yaml 文件不存在，请检查路径 {data_yaml_path} 是否正确")
    else:
        # 进行验证
        results = model.val(data=data_yaml_path, save_json=True)


if __name__ == '__main__':
    # 如果你要将程序冻结成可执行文件，可以添加以下代码，否则可省略
    # freeze_support()
    main()