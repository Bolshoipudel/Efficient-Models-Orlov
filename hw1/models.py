import torch
from torch import nn

CONV_SPECS = [
    (7, 2, 3, 32),
    (5, 1, 32, 64),
    (3, 2, 64, 128),
    (1, 1, 128, 256),
    (3, 2, 256, 256),
    (1, 1, 256, 512),
]
NUM_CLASSES = 100


def conv_bn_relu(k, s, c_in, c_out):
    return [
        nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=k // 2, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    ]


class SmallCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        layers = conv_bn_relu(*CONV_SPECS[0])
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        for spec in CONV_SPECS[1:]:
            layers += conv_bn_relu(*spec)
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(self.pool(x), 1)
        return self.head(x)


def build_model():
    return SmallCNN().eval()


def count_stored_values(model):
    params = sum(p.numel() for p in model.parameters())
    buffers = sum(b.numel() for b in model.buffers() if b.is_floating_point())
    return params + buffers


if __name__ == "__main__":
    model = build_model()
    n = count_stored_values(model)
    print(f"количество параметров: {n:,}  (посчитанные вручную параметры: 1,045,316)  bytes: {4 * n:,}")
    for s in (32, 224):
        with torch.inference_mode():
            out = model(torch.randn(2, 3, s, s))
        print(f"S={s}: output {tuple(out.shape)}")
