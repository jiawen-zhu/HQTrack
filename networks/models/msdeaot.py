import pdb

import torch.nn as nn

from networks.layers.transformer import MSDualBranchGPM
from networks.models.msaot import AOT
from networks.decoders import build_decoder


class MSDeAOT(AOT):
    def __init__(self, cfg, encoder='mobilenetv2', decoder='fpn'):
        super().__init__(cfg, encoder, decoder)

        self.LSTT = MSDualBranchGPM(
            cfg.MODEL_LSTT_NUM,
            cfg.MODEL_ENCODER_EMBEDDING_DIM,
            cfg.MODEL_SELF_HEADS,
            cfg.MODEL_ATT_HEADS,
            emb_dropout=cfg.TRAIN_LSTT_EMB_DROPOUT,
            droppath=cfg.TRAIN_LSTT_DROPPATH,
            lt_dropout=cfg.TRAIN_LSTT_LT_DROPOUT,
            st_dropout=cfg.TRAIN_LSTT_ST_DROPOUT,
            droppath_lst=cfg.TRAIN_LSTT_DROPPATH_LST,
            droppath_scaling=cfg.TRAIN_LSTT_DROPPATH_SCALING,
            intermediate_norm=cfg.MODEL_DECODER_INTERMEDIATE_LSTT,
            return_intermediate=True,
            encoder_dim=cfg.MODEL_ENCODER_DIM)

        decoder_indim = cfg.MODEL_ENCODER_EMBEDDING_DIM * \
            (cfg.MODEL_LSTT_NUM * 2 +
             1) if cfg.MODEL_DECODER_INTERMEDIATE_LSTT else cfg.MODEL_ENCODER_EMBEDDING_DIM * 2

        self.decoder = build_decoder(
            # decoder,
            'fpn2',
            in_dim=decoder_indim,
            out_dim=cfg.MODEL_MAX_OBJ_NUM + 1,
            decode_intermediate_input=cfg.MODEL_DECODER_INTERMEDIATE_LSTT,
            hidden_dim=cfg.MODEL_ENCODER_EMBEDDING_DIM,
            shortcut_dims=cfg.MODEL_ENCODER_DIM,
            align_corners=cfg.MODEL_ALIGN_CORNERS)

        self.id_norm = nn.LayerNorm(cfg.MODEL_ENCODER_EMBEDDING_DIM)
        # self.id_norm_s8 = nn.LayerNorm(128)
        # self.id_norm_s4 = nn.LayerNorm(64)

        self._init_weight()

    def decode_id_logits(self, lstt_emb, shortcuts, step=None):
        # shortcuts from backbone, multi-scale
        n, c, h, w = shortcuts[-1].size()
        decoder_inputs = [shortcuts[-1]]
        for i in range(len(lstt_emb)-1):
            emb=lstt_emb[i]
            decoder_inputs.append(emb.view(h, w, n, -1).permute(2, 3, 0, 1))

        n, c, h, w = shortcuts[-4].size()
        decoder_inputs.append(lstt_emb[-1].view(h, w, n, -1).permute(2, 3, 0, 1))
        pred_logit = self.decoder(decoder_inputs, shortcuts)

        return pred_logit

    def get_id_emb(self, x):
        id_emb = self.patch_wise_id_bank(x)
        id_emb = self.id_norm(id_emb.permute(2, 3, 0, 1)).permute(2, 3, 0, 1)
        id_emb = self.id_dropout(id_emb)

        # id_emb_s8 = self.patch_wise_id_bank_s8(x)
        # id_emb_s8 = self.id_norm_s8(id_emb_s8.permute(2, 3, 0, 1)).permute(2, 3, 0, 1)
        # id_emb_s8 = self.id_dropout(id_emb_s8)

        # id_emb_s4 = self.patch_wise_id_bank_s4(x)
        # id_emb_s4 = self.id_norm_s4(id_emb_s4.permute(2, 3, 0, 1)).permute(2, 3, 0, 1)
        # id_emb_s4 = self.id_dropout(id_emb_s4)

        return id_emb
