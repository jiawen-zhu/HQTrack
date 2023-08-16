import os
import pdb
from .default import DefaultEngineConfig


class EngineConfig(DefaultEngineConfig):
    def __init__(self, exp_name='default', model=''):
        super().__init__(exp_name, model)

        self.STAGE_NAME = 'YTB_DAV_VIP'

        self.init_dir()

        if self.STAGE_NAME == 'YTB_DAV_VIP':
            self.TRAIN_BATCH_SIZE = 1#16  # 16 x3
            self.DATASETS = ['davis2017'] #['vipseg','youtubevos']

            self.DATA_DYNAMIC_MERGE_PROB_VIP = 0.0
            self.DATA_RANDOM_GAP_VIP = 3
            self.DATA_YTB_REPEAT = 1

            pretrain_exp = 'Static_Pre_InternT_MSDeAOTL_V2'
            pretrain_stage = 'PRE'
            pretrain_ckpt = 'save_step_100000.pth'
            self.PRETRAIN = True
            self.PRETRAIN_FULL = True  # if False, load encoder only
            self.PRETRAIN_MODEL = os.path.join(self.DIR_ROOT, 'result',
                                            pretrain_exp, pretrain_stage,
                                            'ema_ckpt', pretrain_ckpt)

            self.TRAIN_TOTAL_STEPS = 150000

        self.TEST_CKPT_PATH = './pretrain_models/temp.pth'

