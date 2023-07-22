from networks.engines.msdeaot_engine_v2 import MSDeAOTEngine_V2, MSDeAOTInferEngine_V2

def build_engine(name, phase='train', **kwargs):
    if name == 'msdeaotengine_v2':
        if phase == 'train':
            return MSDeAOTEngine_V2(**kwargs)
        elif phase == 'eval':
            return MSDeAOTInferEngine_V2(**kwargs)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
