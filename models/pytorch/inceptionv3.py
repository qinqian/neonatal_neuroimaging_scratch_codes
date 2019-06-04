"""models for classification and segmentation
"""
from torch import nn
import torch
from torchsummary import summary
import torch.nn.functional as F


class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, max_pool=True, kernel_size=1, padding=1, stride=1):
        super().__init__()
        block = []
        if max_pool:
            block += [nn.MaxPool3d(kernel_size=3, stride=stride, padding=1)]
        block += [nn.Conv3d(in_channels, out_channels, 
                            kernel_size=kernel_size, padding=padding, stride=stride)]
        block += [nn.ELU()]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super().__init__()
        self.p1   = Conv3dBlock(in_channels, c1, False, 1, 0)

        self.p2_1 = Conv3dBlock(in_channels, c2[0], False, 1, 0)
        self.p2_2 = Conv3dBlock(c2[0], c2[1], False, 3, 1)

        self.p3_1 = Conv3dBlock(in_channels, c3[0], False, 1, 0)
        self.p3_2 = Conv3dBlock(c3[0], c3[1], False, 5, 2, 1)

        self.p4   = Conv3dBlock(in_channels, c4, True, 1, 0, 1)

    def forward(self, x):  # pylint: disable=W0221
        p1 = self.p1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4(x)
        outputs = [p1, p2, p3, p4]
        output = torch.cat(outputs, 1)
        return output


class InceptionAttention3D(nn.Module):
    """ models for classification
    """
    def __init__(self, in_channels, drop=0.5):
        super(InceptionAttention3D, self).__init__()
        self.features = nn.Sequential(
          nn.Conv3d(in_channels, 64, kernel_size=3, stride=2, padding=1),
          nn.ELU(),
          nn.MaxPool3d(3, stride=2, padding=0),
          Inception(64, 64, (96,  128), (16, 32),  32),
          nn.ELU(),
          Inception(64+128+32+32, 128, (128, 192), (32, 96),  64),
          nn.ELU(),
          nn.MaxPool3d(3, stride=2, padding=0),

          Inception(128+192+96+64, 192, (96, 208), (16, 48),   64),
          nn.ELU(),
          Inception(192+208+48+64, 160, (112, 224), (24, 64),  64),
          nn.ELU(),
          Inception(160+224+64+64, 128, (128, 256), (24, 64),  64),
          nn.ELU(),
          Inception(128+256+64+64, 112, (144, 288), (32, 64),  64),
          nn.ELU(),
          #Inception(112+288+64+64, 256,  (160, 320), (32, 128), 128),
          #nn.MaxPool3d(3, stride=2, padding=0),
          #Inception(256+320+128+128, 256,  (160, 320), (32, 128), 128),
          #Inception(256+320+128+128, 384,  (192, 384), (48, 128), 128),
          nn.Dropout(drop),
          nn.AdaptiveAvgPool3d(1)
        )
        #self.fc = nn.Linear(384+384+128+128, 2)
        #self.fc = nn.Linear(128+192+96+64, 2)
        self.fc = nn.Linear(112+288+64+64, 2)

    def forward(self, images):  # pylint: disable=W0221 # ,word_index
        features = self.features(images)
        features = features.view(features.size()[0], -1)
        output = self.fc(features)
        return output


#class InceptionAttention3D2(nn.Module):
#    def __init__(self, in_channels):
#         super().__init__()
#         self.inceptions = InceptionAttention3D(in_channels, 0.2)
#         self.embedding = nn.Embedding(3, 5)
# 
#    def forward(self, images, machine):
#        machine_embed = self.embedding(machine)
#        images_features = self.inceptions(images)

if __name__ == "__main__":
    model = InceptionAttention3D(1).cuda()
    #model = Inception(5, 64, (96,  128), (16, 32),  32).cuda()
    print(summary(model, (1, 87, 112, 93), batch_size=24))
