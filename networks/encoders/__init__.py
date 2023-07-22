import pdb
from networks.encoders.intern_image import INTERN_T,INTERN_H,INTERN_XL,INTERN_B
from networks.layers.normalization import FrozenBatchNorm2d
from torch import nn


def build_encoder(name, frozen_bn=True, freeze_at=-1):
    if frozen_bn:
        BatchNorm = FrozenBatchNorm2d
    else:
        BatchNorm = nn.BatchNorm2d

    if 'intern_t' in name:
        return INTERN_T()
    elif 'intern_h' in name:
        return INTERN_H()
    elif 'intern_xl' in name:
        return INTERN_XL()
    elif 'intern_b' in name:
        return INTERN_B()
    else:
        raise NotImplementedError
