from networks.decoders.fpn import FPNSegmentationHead,FPNSegmentationHead2,FPNSegmentationHead3


def build_decoder(name, **kwargs):

    if name == 'fpn':
        return FPNSegmentationHead(**kwargs)
    elif name == 'fpn2':
        return FPNSegmentationHead2(**kwargs)
    elif name == 'fpn_refine':
        return FPNSegmentationHead3(**kwargs)
    else:
        raise NotImplementedError
