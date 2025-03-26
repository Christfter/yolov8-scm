from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    # model = YOLO('yolov8n.yaml')  # build a new model from YAML
    # model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    # model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
    # model = YOLO('yolov9c.yaml').load('yolov9c.pt')  # build from YAML and transfer weights
    model = YOLO('yolov8-pose-BiFPN-LSKA.yaml')  # build from YAML and transfer weights
    # model = YOLO('yolov8-pose-BiFPN-LSKA.yaml')

    # Train the model
    # model.train(data='D:/YOLOv/datasets/screen/screen.yaml', epochs=10, batch=4, imgsz=640, workers=0)
    model.train(data='coco-pose.yaml', epochs=100, batch=32, imgsz=640, workers=0,project='tomato')

    # model.train(data=r'D:/YOLOv/datasets/red_tomato_gaosi7/data.yaml',
    #             epochs=2,
    #             batch=32,
    #             imgsz=640,
    #             workers=0,
    #             # pretrained = False, #是否在预训练模型权重基础上迁移学习泛华微调
    #             # device = 0, #0表示单卡，0,1,2,3表示多卡，cpu等于使用cpu
    #             project = 'tomato', #项目名称
    #             # optimizer = 'SGD', #默认SGD 备选Adam，AdamW，RMSProp
    #             # cls = 0.5, #目标检测分类损失函数cls_closs权重，默认0.5
    #             # box = 7.5, #目标检测框定位损失函数bbox_loss权重，默认7.5
    #             # dfl = 1.5, #类别不均衡时Dual Focal Loss损失函数dfl_loss权重，默认1.5
    #             # pose = 12.0, #pose关键点定位损失函数pose_loss权重，默认12.0
    #             # kobj = 2.0, #关键点置信度损失函数keypoint_loss权重，默认2.0
    #             )
