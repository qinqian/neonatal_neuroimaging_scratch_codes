"""models for classification and segmentation
"""
from torch import nn
import torch
from torchsummary import summary
##from torchvision.models import inception_v3


class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        block = [nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size)]
        block += [nn.ELU()]
        block += [nn.BatchNorm3d(out_channels)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class InceptionAttention3D(nn.Module):
    """ models for classification
    """

    def __init__(self):
        super(InceptionAttention3D, self).__init__()
        self.conv1 = nn.Conv3d(6, 10, (3, 3, 3))

    def forward(self, images):  # pylint: disable=W0221
        out = self.conv1(images)
        return out


class Inception(nn.Module):
    def __init__(self, c1, c2, c3, c4):
        super().__init__()

        self.p1_1 = nn.Conv3D(c1, kernel_size=1)
        self.p1_1 = nn.ELU(self.p1_1)

        self.p2_1 = nn.Conv3D(c2[0], kernel_size=1, activation=tf.nn.elu)

        self.p2_2 = nn.Conv3D(c2[1], kernel_size=3, padding='SAME', activation=tf.nn.elu)

        self.p3_1 = nn.Conv3D(c3[0], kernel_size=1, activation=tf.nn.elu)
        p3_2 = nn.Conv3D(c3[1], kernel_size=5, padding='SAME', activation=tf.nn.elu)

        p4_1 = nn.MaxPool3d(pool_size=(3, 3, 3), strides=1, padding='SAME')
        p4_2 = nn.Conv3D(c4, kernel_size=1, activation=tf.nn.elu)


    def forward(self, x):  # pylint: disable=W0221
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        return torch.cat([p1, p2, p3, p4], dim=1)


if __name__ == "__main__":
    MODEL = Conv3dBlock(6, 10, 3).cuda()
    print(summary(MODEL, (6, 27, 27, 27), batch_size=24))
