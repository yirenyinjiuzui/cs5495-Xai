# CS5495 — XAI for CNNs

研究问题：**Are the regions/features used by the CNN intuitive? Or are they based on some spurious correlations or noise?**

## 模型和数据下载

完整文件存放在本仓库的 [GitHub Release](https://github.com/yirenyinjiuzui/cs5495-Xai/releases/tag/research-assets-v1)，不放入普通 Git 历史。

| 文件 | 内容 | 大小 |
| --- | --- | --- |
| [resnet18-f37072fd.pth](https://github.com/yirenyinjiuzui/cs5495-Xai/releases/download/research-assets-v1/resnet18-f37072fd.pth) | 普通浮点 ResNet18，`ResNet18_Weights.IMAGENET1K_V1`，1,000 类输出 | 约 47 MB |
| [imagenette2-320.tgz](https://github.com/yirenyinjiuzui/cs5495-Xai/releases/download/research-assets-v1/imagenette2-320.tgz) | fastai Imagenette v2，10 个 ImageNet 类别，短边 320 像素 | 约 342 MB |
| SHA256SUMS.txt | 文件完整性校验值 | 很小 |
| asset_sources.json | 来源、实际字节数、SHA-256、图片数量 | 很小 |

如果 Release 尚未出现，检查 [Publish research assets](https://github.com/yirenyinjiuzui/cs5495-Xai/actions/workflows/publish-assets.yml) 的执行状态。

### 本地使用

建议把权重保存在 `models/resnet18-f37072fd.pth`，将数据解压到 `data/imagenette2-320/`。推荐在 D: 盘保存下载和解压文件。

```python
import torch
from torchvision.models import ResNet18_Weights, resnet18

weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=None)
model.load_state_dict(torch.load(
    "models/resnet18-f37072fd.pth", map_location="cpu", weights_only=True
))
model.eval()
preprocess = weights.transforms()
categories = weights.meta["categories"]
```

模型不重新训练、不量化、不修改分类层。实验保留梯度计算能力；运行 Grad-CAM 等方法时不要把相应前向计算包在 `torch.no_grad()` 中。

Imagenette 的目录标签需要映射到 ImageNet 的 1,000 类索引，不能直接把 `ImageFolder` 产生的 0–9 标签作为模型输出索引。数据集自带的噪声标签 CSV 不用于本项目的主实验。

Imagenette 自身的 train/val 划分不能据此视为与 ResNet18 预训练数据独立的测试集。本项目进行解释性分析，不用它声称模型在全新数据上的泛化表现。主体/背景的精确掩码需要另行准备。

### 从上游重新获取

也可以运行 `python scripts/prepare_assets.py`，把文件下载到本地 `assets/`，校验权重哈希并检查数据压缩包。仅使用 Python 标准库即可执行下载；加载模型另外需要匹配的 PyTorch / Torchvision。

来源及第三方许可说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。项目仓库的许可不改变上游图片或预训练权重的权利归属。
