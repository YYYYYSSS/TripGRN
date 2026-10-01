import torch.nn as nn


class ResBlock1D(nn.Module):
    """残差块（无下采样）"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class ResNetBranch(nn.Module):
    """残差版分支：用展平替代全局池化，并用投影层固定维度。"""

    def __init__(self, input_dim, output_dim=160, fixed_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.fixed_dim = fixed_dim

        self.projection = nn.Linear(input_dim, fixed_dim)

        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.res1 = ResBlock1D(32)
        self.transition = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.res2 = ResBlock1D(64)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * fixed_dim, output_dim),
        )

    def forward(self, x):
        x = self.projection(x)
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.res1(x)
        x = self.transition(x)
        x = self.res2(x)
        return self.fc(x)

