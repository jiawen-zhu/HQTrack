from .default_deaot import DefaultModelConfig


class ModelConfig(DefaultModelConfig):
    def __init__(self):
        super().__init__()
        self.MODEL_NAME = 'InternT_MSDeAOTL_V2'
        self.MODEL_VOS = 'msdeaot_v2'
        self.MODEL_ENGINE = 'msdeaotengine_v2'

        self.MODEL_ENCODER = 'intern_t'
        self.MODEL_ENCODER_DIM = [64, 128, 256, 256]  # 4x, 8x, 16x, 16x
        self.MODEL_LSTT_NUM = 3
        self.TRAIN_LONG_TERM_MEM_GAP = 2

        self.TEST_LONG_TERM_MEM_GAP = 50
        self.TEST_SHORT_TERM_MEM_SKIP = 4
