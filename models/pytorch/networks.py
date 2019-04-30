"""models for classification and segmentation
"""
from torch import nn

class InceptionAttention3D(nn.Module):
    """ models for classification
    """
    def __init__(self):
        super(InceptionAttention3D, self).__init__()
        self.conv1 = nn.Conv1d(6, 10, (3, 3))

    def forward(self, images): # pylint: disable=W0221
        out = self.conv1(images)
        return out

if __name__ == "__main__":
    MODEL = InceptionAttention3D()
    print(MODEL)
