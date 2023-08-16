from .default import DefaultEngineConfig


class EngineConfig(DefaultEngineConfig):
    def __init__(self, exp_name='default', model=''):
        super().__init__(exp_name, model)
        self.STAGE_NAME = 'PRE'

        self.init_dir()

        self.DATASETS = ['static']
        self.TRAIN_TOTAL_STEPS = 100000
        self.TRAIN_BATCH_SIZE = 1 #16

        self.PRETRAIN = True
        self.PRETRAIN_MODEL = './pretrain_models/internimage_t_1k_224_new.pth'

        self.DATA_DYNAMIC_MERGE_PROB = 1.0

        self.TRAIN_LR = 4e-4
        self.TRAIN_LR_MIN = 2e-5
        self.TRAIN_WEIGHT_DECAY = 0.03
        self.TRAIN_SEQ_TRAINING_START_RATIO = 1.0
        self.TRAIN_AUX_LOSS_RATIO = 0.1
