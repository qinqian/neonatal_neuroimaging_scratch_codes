import mxnet as mx
from mxnet import autograd
from mxnet import nd, sym
from mxnet.gluon import nn

class Vgg3d(nn.HybridBlock):
    def __init__(self, conv_arch, **kwargs):
        super(Vgg3d, self).__init__(**kwargs)

        with self.name_scope():
            self.net = nn.HybridSequential()
            for num_convs, num_channels in conv_arch:
                self.net.add(self.get_vgg_block(num_convs, num_channels))
            self.net.add(nn.Dense(100, activation='relu'), 
                         #nn.ELU(alpha=0.3),
                         nn.Dropout(0.5),
                         nn.Dense(2))

    def get_vgg_block(self, num_convs, num_channels):
        blk = nn.HybridSequential()
        for _ in range(num_convs):
            blk.add(nn.Conv3D(num_channels, kernel_size = 3, padding=1, activation='relu'))
            blk.add(nn.BatchNorm())
            #blk.add(nn.ELU(alpha=0.3))
        blk.add(nn.MaxPool3D(pool_size=2, strides=2))
        return blk

    def hybrid_forward(self, F, x):
        return self.net(x)

class ResNext3d(nn.Block):
    def __init__(self, *args, **kwargs):
        super(ResNext3d, self).__init__(*kwargs)
        self.blk = nn.Sequential()
        self.blk.add(nn.Dense(3), nn.Dense(4))
        self.dense = nn.Dense(5)

    def forward(self, x):
        y = nd.relu(self.blk(x))
        print(y)
        return self.dense(y)

n = ResNext3d()
