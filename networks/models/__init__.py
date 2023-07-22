from networks.models.aot import AOT
from networks.models.deaot import DeAOT
from networks.models.msdeaot import MSDeAOT
from networks.models.msdeaot_v2 import MSDeAOT_V2


def build_vos_model(name, cfg, **kwargs):
    if name == 'aot':
        return AOT(cfg, encoder=cfg.MODEL_ENCODER, **kwargs)
    elif name == 'deaot':
        return DeAOT(cfg, encoder=cfg.MODEL_ENCODER, **kwargs)
    elif name == 'msdeaot':
        return MSDeAOT(cfg, encoder=cfg.MODEL_ENCODER, **kwargs)
    elif name == 'msdeaot_v2':
        return MSDeAOT_V2(cfg, encoder=cfg.MODEL_ENCODER, **kwargs)
    else:
        raise NotImplementedError
