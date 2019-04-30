import gluoncv
from gluoncv.utils import LRScheduler
from multiprocessing import cpu_count
from gluoncv.utils.parallel import *
import mxnet as mx
from mxnet.gluon import data as gdata
from mxnet.gluon import data as gdata

DataParallelModel
DataParallelCriterion
kv = mx.kv.create('device')

lr_scheduler = LRScheduler(mode='poly', baselr=0.001, niters=50, nepochs=50)
print(gluoncv.__version__)
CPU_COUNT = cpu_count()
normalize = gdata.vision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_augs = gdata.vision.transforms.Compose([gdata.vision.transforms.RandomResizedCrop(224),
                                              gdata.vision.transforms.RandomFlipLeftRight(),
                                              gdata.vision.transforms.ToTensor(), normalize])

test_augs = gdata.vision.transforms.Compose([gdata.vision.transforms.Resize(256),
                                             gdata.vision.transforms.CenterCrop(224),
                                             gdata.vision.transforms.ToTensor(),
                                             normalize])

def read_image(x):
    import nibabel
    loader = nibabel.load
    image = loader(x)
    return image.get_data()


#model = gluoncv.model_zoo.get_model('fcn_resnet101_voc', pretrained=True)
model =gluoncv.model_zoo.segbase.SegBaseModel
