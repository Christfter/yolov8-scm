# gradcam_compare.py
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# -------------------- 配置参数 --------------------
IMG_PATHS = [
    "./sesame4.jpg",  # 纯芝麻幼苗图片
    "./sesame3.jpg",  # 纯杂草图片
]
MODEL_WEIGHTS = "./runs/detect/train22/weights/best.pt"  # 假设的预训练权重路径
USE_CUDA = True  # 如果没有GPU请设为False


# -------------------- 模型定义 --------------------
class AttentionModule(nn.Module):
    """支持GAP/GMP切换的注意力模块"""

    def __init__(self, in_channels, use_gmp=False):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool2d(1) if use_gmp else nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16),
            nn.ReLU(),
            nn.Linear(in_channels // 16, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SimpleCNN(nn.Module):
    """示例模型结构"""

    def __init__(self, use_gmp=False):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.attention = AttentionModule(64, use_gmp)
        self.classifier = nn.Linear(64, 2)  # 假设2类：芝麻 vs 杂草

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = x.mean([2, 3])  # Global pooling
        return self.classifier(x)


# -------------------- Grad-CAM 工具 --------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册钩子
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x, class_idx=None):
        self.model.eval()
        x = x.requires_grad_(True)

        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1.0
        logits.backward(gradient=one_hot, retain_graph=True)

        pooled_gradients = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        heatmap = torch.mul(self.activations, pooled_gradients).sum(dim=1, keepdim=True)
        heatmap = nn.functional.relu(heatmap)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap.squeeze().cpu().numpy()


# -------------------- 预处理 --------------------
def preprocess(image_path):
    # 读取和预处理
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


# -------------------- 主流程 --------------------
def main():
    # 加载两个对比模型
    model_gap = SimpleCNN(use_gmp=False)
    model_gmp = SimpleCNN(use_gmp=True)

    # 加载预训练权重（这里需要您替换成真实权重）
    # model_gap.load_state_dict(torch.load(MODEL_WEIGHTS))
    # model_gmp.load_state_dict(torch.load(MODEL_WEIGHTS))

    # 遍历所有测试图片
    for img_path in IMG_PATHS:
        # 预处理
        input_tensor = preprocess(img_path)
        original_image = cv2.imread(img_path)
        original_image = cv2.resize(original_image, (224, 224))

        # 生成热力图
        cam_gap = GradCAM(model_gap, model_gap.attention)
        cam_gmp = GradCAM(model_gmp, model_gmp.attention)
        heatmap_gap = cam_gap.generate(input_tensor)
        heatmap_gmp = cam_gmp.generate(input_tensor)

        # 可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original_image)
        axes[0].set_title("Input Image")
        axes[1].imshow(heatmap_gap, cmap='jet')
        axes[1].set_title("GAP Attention")
        axes[2].imshow(heatmap_gmp, cmap='jet')
        axes[2].set_title("GMP Attention")
        plt.suptitle(f"Results for {img_path}", fontsize=14)
        plt.show()


if __name__ == "__main__":
    main()