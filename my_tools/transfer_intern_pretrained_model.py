import pdb
import torch.nn.functional as F
import torch

pretrained_path = './pretrain_models/internimage_b_1k_224.pth'
new_pretrained_path = './pretrain_models/internimage_b_1k_224_new.pth'

ckpt = torch.load(pretrained_path,map_location='cpu')
state_dict = ckpt['model']
for k,v in state_dict.items():
    if 'levels.3' in k:
        if len(v.shape)==1 and 'dcn.mask.bias' not in k and '.dcn.offset.bias' not in k:
            temp = v.unsqueeze(0).unsqueeze(0)
            temp = F.interpolate(temp,scale_factor=0.5,mode='nearest').squeeze(0).squeeze(0)
            state_dict[k]=temp
        if 'dcn.offset.weight' in k or 'dcn.mask.weight' in k:
            temp = v.unsqueeze(0)
            temp = F.interpolate(temp, scale_factor=0.5, mode='nearest').squeeze(0)
            state_dict[k] = temp
        if '_proj.weight' in k or 'mlp.fc1.weight' in k or 'mlp.fc2.weight' in k:
            temp = v.unsqueeze(0).unsqueeze(0)
            temp = F.interpolate(temp, scale_factor=0.5, mode='nearest').squeeze(0).squeeze(0)
            state_dict[k] = temp
        if 'dw_conv.0.weight' in k:
            temp = v[::2,:,:,:]
            state_dict[k] = temp
ckpt['model'] = state_dict
torch.save(ckpt['model'], new_pretrained_path)