# CT 医学图像分割 - Attention Residual 3D UNet

基于 PyTorch 的 3D 医学图像分割项目，使用 Attention Residual U-Net 对 CT 扫描进行分割。

## 功能特性

- **Attention Residual 3D UNet**：结合残差连接与注意力门控，提升特征学习能力
- **自动化预处理**：重采样、方向校正、HU 窗宽窗位处理、基于 Patch 的训练
- **混合精度训练**：使用 AMP 优化 GPU 显存占用
- **多种损失函数**：Dice + BCE 组合损失

## 环境要求

```
torch >= 2.0.0
torchvision
numpy
nibabel >= 4.0.0
scipy
scikit-learn
matplotlib
tqdm
tensorboard
```

安装依赖：

```bash
pip install -r requirements.txt
```

## 数据集结构

```
f:/2.0/
├── pre_ct/                # 原始 CT 扫描（NIfTI 格式）
│   ├── 77__Se301__Shoulder 0.625mm STND__17562306.nii.gz
│   ├── 78__Se301__Shoulder 0.625mm STND__...
│   └── ...
├── pre_ct_seg/           # 分割标签（NIfTI 格式）
│   ├── 77_seg.nii.gz
│   ├── 78_seg.nii.gz
│   └── ...
└── src/                  # 源代码
```

## 使用方法

### 训练模型

```bash
python src/train.py
```

自定义参数训练：

```bash
python src/train.py --epochs 50 --batch_size 4 --lr 1e-4
```

### 推理预测

```bash
python src/inference.py \
    --checkpoint outputs/best_model.pth \
    --input pre_ct/100__Se301__Shoulder\ 0.625mm\ STND__01082467.nii.gz \
    --output outputs/predictions
```

## 配置参数

编辑 `src/config.py` 自定义训练：

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `TRAIN_IDS` | 77-117 | 训练样本 ID |
| `BATCH_SIZE` | 2 | 批次大小 |
| `NUM_EPOCHS` | 100 | 训练轮数 |
| `LEARNING_RATE` | 1e-4 | 学习率 |
| `PATCH_SIZE` | (128, 128, 128) | 输入 Patch 尺寸 |
| `TARGET_SPACING` | [1.0, 1.0, 1.0] | 目标体素间距（mm） |
| `HU_WINDOW_MIN` | -300 | HU 窗宽最小值 |
| `HU_WINDOW_MAX` | 1200 | HU 窗宽最大值 |

## 项目结构

```
src/
├── config.py              # 配置文件
├── train.py               # 训练脚本
├── inference.py           # 推理脚本
├── requirements.txt       # 依赖列表
├── models/
│   └── attention_res_unet3d.py   # Attention Residual 3D UNet 模型
├── data/
│   └── dataset.py         # 数据加载与预处理
└── utils/
    ├── losses.py          # 损失函数（Dice、BCE、Focal）
    └── metrics.py         # 评估指标（Dice、IoU、F1）
```

## 模型架构

**Attention Residual 3D UNet** 结合了以下技术：

1. **残差连接**：缓解梯度消失，支持更深网络
2. **注意力门控**：聚焦相关特征，抑制无关区域
3. **U-Net 结构**：编码器-解码器 + 跳跃连接，提取多尺度特征

架构细节：
- 4 级编码器-解码器
- 特征通道数：32 → 64 → 128 → 256
- 3D 卷积 + BatchNorm + ReLU
- 跳跃连接上加入注意力门控

## 预处理流程

1. **方向校正**：NIfTI 转换为 RAS+ 方向
2. **重采样**：所有体素重采样至 1.0×1.0×1.0 mm
3. **HU 窗宽裁剪**：CT 值裁剪至骨窗 [-300, 1200] HU
4. **D 维度裁剪**：中心裁剪至 392 层，覆盖所有标注区域
5. **Patch 提取**：训练时随机裁剪 128×128×128 的 Patch
6. **Z-Score 归一化**：零均值、单位方差

## 模型输出

训练输出保存在 `outputs/` 目录：

```
outputs/
├── best_model.pth        # 最佳模型检查点
└── checkpoint_epoch_*.pth # 周期性检查点
```