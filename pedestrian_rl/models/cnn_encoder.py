import torch
import torch.nn as nn
from ..utils.config_loader import load_config


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class CNNEncoder(nn.Module):
    """
    CNN encoder for pedestrian BEV.
    Input:
        (H, W, C) or (B, H, W, C) or (B, C, H, W)
    Output:
        (B, feature_dim)
    """
    cnn_config = load_config("training_config.json")["cnn"]

    def __init__(self, 
                 input_channels=cnn_config["input_channels"], 
                 feature_dim=cnn_config["bev_feature_dim"]
                ):
        super().__init__()

        # stem: 160 -> 80
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 80x80
        self.stage1 = ResidualBlock(32, 32, stride=1)

        # 80 -> 40
        self.stage2 = ResidualBlock(32, 64, stride=2)

        # 40x40
        self.stage3 = ResidualBlock(64, 64, stride=1)

        # 40 -> 20
        self.stage4 = ResidualBlock(64, 128, stride=2)

        # 20x20
        self.stage5 = ResidualBlock(128, 128, stride=1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True)
        )

    def _preprocess_input(self, bev_data):
        if not isinstance(bev_data, torch.Tensor):
            bev_data = torch.tensor(bev_data, dtype=torch.float32)
        else:
            bev_data = bev_data.float()

        if bev_data.max() > 1.0:
            bev_data = bev_data / 255.0

        if bev_data.dim() == 3:
            bev_data = bev_data.unsqueeze(0)

        # (B, H, W, C) -> (B, C, H, W)
        if bev_data.dim() == 4 and bev_data.shape[-1] <= 16:
            bev_data = bev_data.permute(0, 3, 1, 2).contiguous()

        if bev_data.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got shape {bev_data.shape}")

        return bev_data

    def forward(self, bev_data):
        x = self._preprocess_input(bev_data)

        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        x = self.pool(x)
        x = self.proj(x)
        return x




