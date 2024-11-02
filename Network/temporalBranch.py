import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, g_outputs, targets, d_outputs, lambda_adv=0.001):
        loss1 = F.mse_loss(g_outputs, targets)
        loss2 = lambda_adv * torch.log(d_outputs + 1e-8)
        loss = loss1 - loss2
        loss = loss.mean()
        return loss


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, d_outputs_real, d_outputs_fake):
        loss1 = -torch.log(d_outputs_real + 1e-8)
        loss2 = -torch.log(1 - d_outputs_fake + 1e-8)
        loss = loss1 + loss2
        loss = loss.mean()
        return loss


class Res1(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 7, 7),
            stride=(1, 2, 2),
            padding=(0, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class Res2_5_basic(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Deconv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.trsnspose = nn.ConvTranspose2d(
            out_channels,
            out_channels,
            kernel_size=4,
            stride=stride,
            padding=padding,
            bias=False,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.trsnspose(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.res1 = Res1()
        self.res2 = self._make_res2_5(in_channels=64, out_channels=64, stride=1)
        self.res3 = self._make_res2_5(in_channels=64, out_channels=128, stride=2)
        self.res4 = self._make_res2_5(in_channels=128, out_channels=256, stride=2)
        self.res5 = self._make_res2_5(in_channels=256, out_channels=512, stride=2)
        self.deconv1 = Deconv(in_channels=512, out_channels=256, stride=2, padding=1)
        self.deconv2 = Deconv(in_channels=512, out_channels=128, stride=2, padding=1)
        self.deconv3 = Deconv(in_channels=256, out_channels=64, stride=2, padding=1)
        self.deconv4 = Deconv(in_channels=64, out_channels=32, stride=4, padding=0)
        self.convout = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)

    def _make_res2_5(self, in_channels, out_channels, stride):
        return nn.Sequential(
            Res2_5_basic(in_channels, out_channels, stride),
            Res2_5_basic(out_channels, out_channels, 1),
        )

    def _timeMaxPool(self, x, depth):
        x = nn.MaxPool3d(
            kernel_size=(depth, 1, 1),
            stride=(depth, 1, 1),
        )(x)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[3], x.shape[4])
        return x

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)

        x = self.res3(x)
        t1 = self._timeMaxPool(x, x.shape[2])

        x = self.res4(x)
        t2 = self._timeMaxPool(x, x.shape[2])

        x = self.res5(x)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[3], x.shape[4])

        x = self.deconv1(x)
        x = torch.cat((x, t2), dim=1)

        x = self.deconv2(x)
        x = torch.cat((x, t1), dim=1)

        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.convout(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.inception_v3 = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT
        )
        self.inception_v3.aux_logits = False
        self.inception_v3.AuxLogits = None
        self.inception_v3.fc = nn.Linear(self.inception_v3.fc.in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.inception_v3(x)
        x = self.sigmoid(x)
        return x


if __name__ == "__main__":
    model = Generator()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    x = torch.randn(1, 3, 8, 1024, 1024).to(device)
    y = model(x)
