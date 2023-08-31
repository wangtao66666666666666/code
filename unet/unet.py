from torch import nn
import torch
from collections import OrderedDict
from torchsummary import summary

class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        # 编码器
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

       # bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

       # 解码器
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)  # 那根线
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels, # 确定卷积核的深度
                            out_channels=features, # 确实输出的特征图深度，即卷积核组的多少
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


# class ConvBlock(nn.Module):
#     """ implement conv+ReLU two times """
#     def __init__(self, in_channels, middle_channels, out_channels):
#         super().__init__()
#         conv_relu = []
#         conv_relu.append(nn.Conv2d(in_channels=in_channels, out_channels=middle_channels,
#                                    kernel_size=3, padding=1, stride=1))
#         conv_relu.append(nn.ReLU())
#         conv_relu.append(nn.Conv2d(in_channels=middle_channels, out_channels=out_channels,
#                                    kernel_size=3, padding=1, stride=1))
#         conv_relu.append(nn.ReLU())
#         self.conv_ReLU = nn.Sequential(*conv_relu)
#     def forward(self, x):
#         out = self.conv_ReLU(x)
#         return out
#
# class U_Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         # 首先定义左半部分网络
#         # left_conv_1 表示连续的两个（卷积+激活）
#         # 随后进行最大池化
#         self.left_conv_1 = ConvBlock(in_channels=1, middle_channels=64, out_channels=64)
#         self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.left_conv_2 = ConvBlock(in_channels=64, middle_channels=128, out_channels=128)
#         self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.left_conv_3 = ConvBlock(in_channels=128, middle_channels=256, out_channels=256)
#         self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.left_conv_4 = ConvBlock(in_channels=256, middle_channels=512, out_channels=512)
#         self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.left_conv_5 = ConvBlock(in_channels=512, middle_channels=1024, out_channels=1024)
#
#         # 定义右半部分网络
#         self.deconv_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.right_conv_1 = ConvBlock(in_channels=1024, middle_channels=512, out_channels=512)
#
#         self.deconv_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=2, output_padding=1)
#         self.right_conv_2 = ConvBlock(in_channels=512, middle_channels=256, out_channels=256)
#
#         self.deconv_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=2 ,output_padding=1)
#         self.right_conv_3 = ConvBlock(in_channels=256, middle_channels=128, out_channels=128)
#
#         self.deconv_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, output_padding=1, padding=1)
#         self.right_conv_4 = ConvBlock(in_channels=128, middle_channels=64, out_channels=64)
#         # 最后是1x1的卷积，用于将通道数化为3
#         self.right_conv_5 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#
#         # 1：进行编码过程
#         feature_1 = self.left_conv_1(x)
#         feature_1_pool = self.pool_1(feature_1)
#
#         feature_2 = self.left_conv_2(feature_1_pool)
#         feature_2_pool = self.pool_2(feature_2)
#
#         feature_3 = self.left_conv_3(feature_2_pool)
#         feature_3_pool = self.pool_3(feature_3)
#
#         feature_4 = self.left_conv_4(feature_3_pool)
#         feature_4_pool = self.pool_4(feature_4)
#
#         feature_5 = self.left_conv_5(feature_4_pool)
#
#         # 2：进行解码过程
#         de_feature_1 = self.deconv_1(feature_5)
#         # 特征拼接
#         temp = torch.cat((feature_4, de_feature_1), dim=1)
#         de_feature_1_conv = self.right_conv_1(temp)
#
#         de_feature_2 = self.deconv_2(de_feature_1_conv)
#         temp = torch.cat((feature_3, de_feature_2), dim=1)
#         de_feature_2_conv = self.right_conv_2(temp)
#
#         de_feature_3 = self.deconv_3(de_feature_2_conv)
#
#         temp = torch.cat((feature_2, de_feature_3), dim=1)
#         de_feature_3_conv = self.right_conv_3(temp)
#
#         de_feature_4 = self.deconv_4(de_feature_3_conv)
#         temp = torch.cat((feature_1, de_feature_4), dim=1)
#         de_feature_4_conv = self.right_conv_4(temp)
#
#         out = self.right_conv_5(de_feature_4_conv)
#
#         return out
#
# if __name__ == "__main__":
#     x = torch.rand(size=(1,1,128,128))
#     net = UNet()
# #     out = net(x)
# #     print(out.size())
# #     print("ok")
# #     print(x)
# #     print(out)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = UNet().to(device)
#     summary(model, input_size=(1, 128, 128))
