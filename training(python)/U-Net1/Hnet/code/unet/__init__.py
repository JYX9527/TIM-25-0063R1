"""
这是根据UNet模型搭建出的一个基本网络结构
输入和输出大小是一样的，可以根据需求进行修改
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary


# 基本卷积块
class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(C_out, C_out, 3, 1, 1),
        )
        self.layer3= nn.Sequential(
            nn.ReLU(),
        )
    def forward(self, x):
        y=self.layer1(x)
        identity=y
        z=self.layer2(y)
        a=identity+z
        return self.layer3(a)


# 下采样模块
class DownSampling(nn.Module):
    def __init__(self):
        super(DownSampling, self).__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(kernel_size=2)
        )
    def forward(self, x):
        return self.layer(x)


# 上采样模块
class UpSampling(nn.Module):
    def __init__(self, C):
        super(UpSampling, self).__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2,mode="nearest"),
            nn.Conv2d(C, C // 2, 3, 1, 1),
            nn.ReLU(),
        )
    def forward(self, x, r):
        x = self.layer(x)
        return torch.cat((x, r), 1)




# 主干网络
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()


        self.C1 = Conv(1, 64)
        self.C2 = Conv(64, 128)
        self.C3 = Conv(128, 256)
        self.C4 = Conv(256, 512)

        self.C5 = Conv(512, 1024)

        self.C6 = Conv(1024, 512)
        self.C7 = Conv(512, 256)
        self.C8 = Conv(256, 128)
        self.C9 = Conv(128, 64)

        self.U1 = UpSampling(1024)
        self.U2 = UpSampling(512)
        self.U3 = UpSampling(256)
        self.U4 = UpSampling(128)
        self.D=DownSampling()


        self.pred = torch.nn.Conv2d(64,2, 3, 1, 1)

    def forward(self, x):
        # 下采样部分
        R1 = self.C1(x)
        R2 = self.C2(self.D(R1))
        R3 = self.C3(self.D(R2))
        R4 = self.C4(self.D(R3))


        X= self.C5(self.D(R4))

        # 上采样部分
        Y4 = self.C6(self.U1(X, R4))
        Y3 = self.C7(self.U2(Y4, R3))
        Y2 = self.C8(self.U3(Y3, R2))
        Y1 = self.C9(self.U4(Y2, R1))

        return self.pred(Y1)

if __name__ == '__main__':
    # a=torch.randn(1,1,640,480)
    net = UNet()
    def initialize_weights(model):
        for m in model.modules():
            if isinstance(m,nn.Conv2d):
                 a = m.weight
                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    initialize_weights(net)
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            a = m.weight
    net = UNet().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # a=a.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # b=net(a)

    summary(net, (1,640, 480))