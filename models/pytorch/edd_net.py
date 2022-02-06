from torch import nn
from torchsummary import summary
import torch


class n_conv(nn.Module):
    def __init__(self, in_ch, out_ch, conv_num=2):
        super(n_conv, self).__init__()

        self.convs = []
        self.convs.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.convs.append(nn.ELU(inplace=True))
        for n in range(conv_num - 1):
            self.convs.append(nn.Conv2d(out_ch, out_ch, 3, padding=1))
            self.convs.append(nn.ELU(inplace=True))
        self.convs = nn.Sequential(*self.convs)

    def forward(self, x):
        x = self.convs(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, conv_num=2, pool=True):
        super(down, self).__init__()
        self.nconv = n_conv(in_ch, out_ch, conv_num)
        self.maxpool = nn.MaxPool2d((2, 2), stride=2)
        self.pool = pool

    def forward(self, x):
        x = self.nconv(x)
        if self.pool:
            x = self.maxpool(x)
        return x


class n_deconv(nn.Module):
    def __init__(self, n_in, n_out, deconv_num=5):
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        super(n_deconv, self).__init__()
        self.deconvs = []
        self.deconvs.append(nn.ConvTranspose2d(n_in, n_out, 3, stride=1, padding=1))
        self.deconvs.append(nn.ELU(inplace=True))
        for n in range(deconv_num):
            self.deconvs.append(nn.ConvTranspose2d(n_out, n_out, 3, stride=1, padding=1))
            self.deconvs.append(nn.ELU(inplace=True))

        self.deconvs = nn.Sequential(*self.deconvs)

    def forward(self, x):
        x = self.deconvs(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, deconv_num, up_sample):
        super(up, self).__init__()
        self.up_sample = up_sample
        self.conv = n_deconv(in_ch, out_ch, deconv_num)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv(x)
        if self.up_sample:
            x = self.up(x)
        return x


class EDDNet(nn.Module):
    def __init__(self, n_in, n_out, n_deconv=2):
        super(EDDNet, self).__init__()
        self.n_out = n_out

        self.deconv_blocks = nn.ModuleList()
        for i in range(n_deconv):
            down1 = down(n_in, 32, 2, pool=True)
            #down2 = down(32, 64, 2, pool=True)
            #down3 = down(64, 128, 3, pool=False)
            down2 = down(32, 64, 4, pool=True)
            down3 = down(64, 128, 5, pool=False)

            up1 = up(128, 128, 6, True)
            up2 = up(128, 64, 3, False)
            #up1 = up(128, 128, 3, True)
            #up2 = up(128, 64, 2, False)

            up3 = up(64, 32, 1, True)
            up4 = up(32, 32, 1, False)
            up5 = up(32, 2, 1, False)

            deconv = nn.Sequential(
                down1,
                down2,
                down3,

                up1,
                up2,
                up3,
                up4,
                up5
            )
            self.deconv_blocks.append(deconv)
        self.down = down(2, 32, 2, pool=False)
        self.conv = down(32, n_out, 1, pool=False)

    def forward(self, x):
        res = [ d(x) for d in self.deconv_blocks ]
        y = torch.cat(res, 1)
        y = torch.sum(y, 1, keepdim=True)
        y = torch.cat([y, x], dim=1)
        y = self.down(y)
        y = self.conv(y)
        y = torch.sigmoid(y)
        return y


if __name__ == '__main__':
    net = EDDNet(1, 2, 2)
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #net = net.to(device)
    net = net.cuda()

    print(summary(net, (1, 64, 64), batch_size=4))
#    test = torch.randn([32, 1, 64, 64]).to(device)
#    print(net(test).shape)
