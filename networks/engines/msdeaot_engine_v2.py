import pdb
import math
import numpy as np

from utils.image import one_hot_mask

from networks.layers.basic import seq_to_2d
from networks.engines.msaot_engine_v2 import AOTEngine, AOTInferEngine


class MSDeAOTEngine_V2(AOTEngine):
    def __init__(self,
                 aot_model,
                 gpu_id=0,
                 long_term_mem_gap=9999,
                 short_term_mem_skip=1,
                 layer_loss_scaling_ratio=2.):
        super().__init__(aot_model, gpu_id, long_term_mem_gap,
                         short_term_mem_skip)
        self.layer_loss_scaling_ratio = layer_loss_scaling_ratio

    def update_short_term_memory(self, curr_mask, curr_id_emb=None):
        if curr_id_emb is None:
            if len(curr_mask.size()) == 3 or curr_mask.size()[0] == 1:
                curr_one_hot_mask = one_hot_mask(curr_mask, self.max_obj_num)
            else:
                curr_one_hot_mask = curr_mask
            curr_id_emb = self.assign_identity(curr_one_hot_mask)

        lstt_curr_memories = self.curr_lstt_output[1]
        lstt_curr_memories_2d = []

        for layer_idx in range(len(lstt_curr_memories)):
            curr_k, curr_v, curr_id_k, curr_id_v = lstt_curr_memories[
                layer_idx]
            if layer_idx in [0,1,2]:
                curr_id_k, curr_id_v = self.AOT.LSTT.layers[
                    layer_idx].fuse_key_value_id(curr_id_k, curr_id_v, curr_id_emb[0])
                lstt_curr_memories[layer_idx][2], lstt_curr_memories[layer_idx][3] = curr_id_k, curr_id_v
                local_curr_id_k = seq_to_2d(
                    curr_id_k, self.enc_size_2d) if curr_id_k is not None else None
                local_curr_id_v = seq_to_2d(curr_id_v, self.enc_size_2d)
                lstt_curr_memories_2d.append([
                    seq_to_2d(curr_k, self.enc_size_2d),
                    seq_to_2d(curr_v, self.enc_size_2d), local_curr_id_k,
                    local_curr_id_v
                ])
            else:
                curr_id_k, curr_id_v = self.AOT.LSTT.layer_s8.fuse_key_value_id(curr_id_k, curr_id_v, curr_id_emb[1])
                lstt_curr_memories[layer_idx][2], lstt_curr_memories[layer_idx][3] = curr_id_k, curr_id_v
                s8_size = [math.ceil(curr_mask.shape[-2] / 8), math.ceil(curr_mask.shape[-1] / 8)]
                local_curr_id_k = seq_to_2d(
                    curr_id_k,(s8_size[0],s8_size[1])) if curr_id_k is not None else None
                local_curr_id_v = seq_to_2d(curr_id_v,  (s8_size[0],s8_size[1]))
                lstt_curr_memories_2d.append([
                    seq_to_2d(curr_k, (s8_size[0],s8_size[1])),
                    seq_to_2d(curr_v,(s8_size[0],s8_size[1])), local_curr_id_k,
                    local_curr_id_v
                ])

        self.short_term_memories_list.append(lstt_curr_memories_2d)
        self.short_term_memories_list = self.short_term_memories_list[
            -self.short_term_mem_skip:]
        self.short_term_memories = self.short_term_memories_list[0]

        if self.frame_step - self.last_mem_step >= self.long_term_mem_gap:
            self.update_long_term_memory(lstt_curr_memories)
            self.last_mem_step = self.frame_step


class MSDeAOTInferEngine_V2(AOTInferEngine):
    def __init__(self,
                 aot_model,
                 gpu_id=0,
                 long_term_mem_gap=9999,
                 short_term_mem_skip=1,
                 max_aot_obj_num=None):
        super().__init__(aot_model, gpu_id, long_term_mem_gap,
                         short_term_mem_skip, max_aot_obj_num)

    def add_reference_frame(self, img, mask, obj_nums, frame_step=-1):
        if isinstance(obj_nums, list):
            obj_nums = obj_nums[0]
        self.obj_nums = obj_nums
        aot_num = max(np.ceil(obj_nums / self.max_aot_obj_num), 1)
        while (aot_num > len(self.aot_engines)):
            new_engine = MSDeAOTEngine_V2(self.AOT, self.gpu_id,
                                     self.long_term_mem_gap,
                                     self.short_term_mem_skip)
            new_engine.eval()
            self.aot_engines.append(new_engine)

        separated_masks, separated_obj_nums = self.separate_mask(
            mask, obj_nums)
        img_embs = None

        for aot_engine, separated_mask, separated_obj_num in zip(
                self.aot_engines, separated_masks, separated_obj_nums):

            aot_engine.add_reference_frame(img,
                                           separated_mask,
                                           obj_nums=[separated_obj_num],
                                           frame_step=frame_step,
                                           img_embs=img_embs)
            if img_embs is None: 
                img_embs = aot_engine.curr_enc_embs

        self.update_size()
