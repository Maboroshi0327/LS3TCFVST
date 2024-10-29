import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

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


class ResNet(nn.Module):

    def __init__(
        self,
        block,
        layers,
        block_inplanes,
        n_input_channels=3,
        conv1_t_size=7,
        conv1_t_stride=1,
        no_max_pool=False,
        shortcut_type="B",
        widen_factor=1.0,
    ):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(
            n_input_channels,
            self.in_planes,
            kernel_size=(conv1_t_size, 7, 7),
            stride=(conv1_t_stride, 2, 2),
            padding=(conv1_t_size // 2, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, block_inplanes[0], layers[0], shortcut_type
        )
        self.layer2 = self._make_layer(
            block, block_inplanes[1], layers[1], shortcut_type, stride=2
        )
        self.layer3 = self._make_layer(
            block, block_inplanes[2], layers[2], shortcut_type, stride=2
        )
        self.layer4 = self._make_layer(
            block, block_inplanes[3], layers[3], shortcut_type, stride=2
        )

        self.deconv1 = Deconv(in_channels=512, out_channels=256, stride=2, padding=1)
        self.deconv2 = Deconv(in_channels=512, out_channels=128, stride=2, padding=1)
        self.deconv3 = Deconv(in_channels=256, out_channels=64, stride=2, padding=1)
        self.deconv4 = Deconv(in_channels=64, out_channels=32, stride=4, padding=0)
        self.convout = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(
            out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)
        )
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    self._downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []
        layers.append(
            block(
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
                downsample=downsample,
            )
        )
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _timeMaxPool(self, x, depth):
        x = nn.MaxPool3d(
            kernel_size=(depth, 1, 1),
            stride=(depth, 1, 1),
        )(x)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[3], x.shape[4])
        return x

    def forward(self, x):
        # Res1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        # Res2
        x = self.layer1(x)

        # Res3
        x = self.layer2(x)
        t1 = self._timeMaxPool(x, x.shape[2])

        # Res4
        x = self.layer3(x)
        t2 = self._timeMaxPool(x, x.shape[2])

        # Res5
        x = self.layer4(x)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[3], x.shape[4])

        # Deconv1
        x = self.deconv1(x)
        x = torch.cat((x, t2), dim=1)

        # Deconv2
        x = self.deconv2(x)
        x = torch.cat((x, t1), dim=1)

        # Deconv3~4 and ConvOut
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.convout(x)

        return x


def generate_model(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    return model


def Generator():
    model = generate_model(
        n_input_channels=3,
        shortcut_type="B",
        conv1_t_size=7,
        conv1_t_stride=1,
        no_max_pool=False,
        widen_factor=1.0,
    )

    # 加載已訓練模型的參數
    trained_model_path = "./Pretrained/r3d18_K_200ep.pth"
    trained_state_dict = torch.load(trained_model_path, weights_only=False)

    # 只加載模型中已有的參數
    new_state_dict = {
        k: v
        for k, v in trained_state_dict["state_dict"].items()
        if k in model.state_dict()
    }
    model.load_state_dict(new_state_dict, strict=False)

    # 檢查新的模型是否正確加載了參數
    for name, param in model.named_parameters():
        if name in new_state_dict:
            print(f"Loaded {name} from trained model.")
        else:
            print(f"Initialized {name} randomly.")

    return model


if __name__ == "__main__":
    # 建立模型
    model = Generator()

    # 測試模型
    x = torch.randn(1, 3, 8, 512, 512)
    print("Input size: ", x.size())
    y = model(x)
    print("Output size: ", y.size())
