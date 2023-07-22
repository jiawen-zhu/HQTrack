import pdb
from random import sample
from PIL import Image
from rsa import sign
import torch
import torch.nn.functional as F
import os
import sys
import cv2
import importlib
import numpy as np
import math
import random
from torchvision import transforms
import time
# import segmentation_refinement as refine
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor



sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import vots_utils
from utils.image import flip_tensor
AOT_PATH = os.path.join(os.path.dirname(__file__), '..')
import dataloaders.video_transforms as tr

from networks.engines import build_engine
from utils.checkpoint import load_network
from networks.models import build_vos_model
from utils.metric import pytorch_iou

# from MS_AOT.MixFormer.lib.test.tracker.mixformer_online import MixFormerOnline
# import MS_AOT.MixFormer.lib.test.parameter.mixformer_online as vot_params

cur_colors = [(0, 255, 255), # yellow b g r
              (255, 0, 0), # blue
              (0, 255, 0), # green
              (0, 0, 255), # red
              (255, 255, 255), # white
              (0, 0, 0), # black
              (255, 255, 0), # Cyan
              (225, 228, 255), # MistyRose
              (180, 105, 255), # HotPink
              (255, 0, 255), # Magenta
              ]*100

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


seed_torch(1000000007)
torch.set_num_threads(4)
torch.autograd.set_grad_enabled(False)


class AOTTracker(object):
    def __init__(self, cfg, gpu_id):
        self.with_crop = False  
        self.EXPAND_SCALE = None
        self.small_ratio = 12
        self.mid_ratio = 100
        self.large_ratio = 0.5
        self.AOT_INPUT_SIZE = (465, 465) 
        self.cnt = 2
        self.gpu_id = gpu_id
        self.model = build_vos_model(cfg.MODEL_VOS, cfg).cuda(gpu_id)
        # self.model = build_vos_model(cfg.MODEL_VOS, cfg).half()
        self.model.cuda(gpu_id)
        self.model.eval() 
        print('cfg.TEST_CKPT_PATH = ', cfg.TEST_CKPT_PATH)
        self.model, _ = load_network(self.model, cfg.TEST_CKPT_PATH, gpu_id)
        self.aug_nums = len(cfg.TEST_MULTISCALE)
        if cfg.TEST_FLIP:
            self.aug_nums*=2
        self.engine = []
        for aug_idx in range(self.aug_nums):
            self.engine.append(build_engine(cfg.MODEL_ENGINE,
                                   phase='eval',
                                   aot_model=self.model,
                                   gpu_id=gpu_id,
                                   short_term_mem_skip=cfg.TEST_SHORT_TERM_MEM_SKIP,
                                   long_term_mem_gap=cfg.TEST_LONG_TERM_MEM_GAP,
                                ))
            self.engine[-1].eval()
        self.transform = transforms.Compose([
        tr.MultiRestrictSize_(cfg.TEST_MAX_SHORT_EDGE,
                                cfg.TEST_MAX_LONG_EDGE, cfg.TEST_FLIP, cfg.TEST_INPLACE_FLIP,
                                cfg.TEST_MULTISCALE, cfg.MODEL_ALIGN_CORNERS),
        tr.MultiToTensor()
        ])  

    def add_first_frame(self, frame, mask):
        sample = {
            'current_img': frame,
            'current_label': mask,
            'height':frame.shape[0],
            'weight':frame.shape[1]
        }
        sample = self.transform(sample)

        if self.aug_nums>1:
            torch.cuda.empty_cache()  
        for aug_idx in range(self.aug_nums):
            frame = sample[aug_idx]['current_img'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)
            mask = sample[aug_idx]['current_label'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)
            mask = F.interpolate(mask, size=frame.size()[2:], mode="nearest")
            self.engine[aug_idx].add_reference_frame(frame, mask, frame_step=0, obj_nums=int(mask.max()))

    def track(self, image):
        
        height = image.shape[0]
        width = image.shape[1]
        sample = {'current_img': image}
        sample['meta'] = {
            'height': height,
            'width': width,
            'flip': False
        }
        sample = self.transform(sample)


        if self.aug_nums>1:
            torch.cuda.empty_cache()
        all_preds = []
        for aug_idx in range(self.aug_nums):
            output_height = sample[aug_idx]['meta']['height']
            output_width = sample[aug_idx]['meta']['width']
            image = sample[aug_idx]['current_img'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)
            image = image.cuda(self.gpu_id, non_blocking=True)
            self.engine[aug_idx].match_propogate_one_frame(image)
            is_flipped = sample[aug_idx]['meta']['flip']
            pred_logit = self.engine[aug_idx].decode_current_logits((output_height, output_width))
            if is_flipped:
                pred_logit = flip_tensor(pred_logit, 3)
            pred_prob = torch.softmax(pred_logit, dim=1)
            all_preds.append(pred_prob)
            cat_all_preds = torch.cat(all_preds, dim=0)
            pred_prob = torch.mean(cat_all_preds,dim=0,keepdim=True)
            pred_label = torch.argmax(pred_prob,dim=1,keepdim=True).float()
            _pred_label = F.interpolate(pred_label,
                                        size=self.engine[aug_idx].input_size_2d,
                                        mode="nearest")
            self.engine[aug_idx].update_memory(_pred_label)
            mask = pred_label.detach().cpu().numpy()[0][0].astype(np.uint8)
        conf = 0

        return mask, conf


def make_full_size(x, output_sz):
    '''
    zero-pad input x (right and down) to match output_sz
    x: numpy array e.g., binary mask
    output_sz: size of the output [width, height]
    '''
    if x.shape[0] == output_sz[1] and x.shape[1] == output_sz[0]:
        return x
    pad_x = output_sz[0] - x.shape[1]
    if pad_x < 0:
        x = x[:, :x.shape[1] + pad_x]
        # padding has to be set to zero, otherwise pad function fails
        pad_x = 0
    pad_y = output_sz[1] - x.shape[0]
    if pad_y < 0:
        x = x[:x.shape[0] + pad_y, :]
        # padding has to be set to zero, otherwise pad function fails
        pad_y = 0
    return np.pad(x, ((0, pad_y), (0, pad_x)), 'constant', constant_values=0)


def read_img(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _rect_from_mask(mask):
    if len(np.where(mask==1)[0]) == 0:
        return None
    x_ = np.sum(mask, axis=0)
    y_ = np.sum(mask, axis=1)
    x0 = np.min(np.nonzero(x_))
    x1 = np.max(np.nonzero(x_))
    y0 = np.min(np.nonzero(y_))
    y1 = np.max(np.nonzero(y_))
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def select_tracker(img, mask):
    img_sz = img.shape[0] * img.shape[1]
    _, _, w, h = _rect_from_mask(mask)
    max_edge = max(w, h)
    rect_sz = max_edge * max_edge
    ratio = img_sz / rect_sz
    print("ratio = {ratio}")
    if ratio > 900:
        return "aot_mix"
    else:
        return "aot"


class MSAOTTracker(object):
    def __init__(self, cfg, config, local_track=False,sam_refine=False,sam_refine_iou=0):
        self.mask_size = None
        self.local_track = local_track
        self.aot_tracker = AOTTracker(cfg, config['gpu_id'])
        # SAM
        self.sam_refine=sam_refine
        if self.sam_refine:
            model_type = 'vit_h' #'vit_h'
            sam_checkpoint = os.path.join(os.path.dirname(__file__), '..', 'segment_anything_hq/pretrained_model/sam_hq_vit_h.pth')
            output_mode = "binary_mask"
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=torch.device('cuda'))
            self.mask_generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode)
            self.mask_prompt = SamPredictor(sam)
        self.sam_refine_iou=sam_refine_iou

    def get_box(self, label):
        thre = np.max(label) * 0.5
        label[label > thre] = 1
        label[label <= thre] = 0
        a = np.where(label != 0)
        height, width = label.shape
        ratio = 0.0

        if len(a[0]) != 0:
            bbox1 = np.stack([np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0])])
            w, h = np.max(a[1]) - np.min(a[1]), np.max(a[0]) - np.min(a[0])
            x1 = max(bbox1[0] - w * ratio, 0)
            y1 = max(bbox1[1] - h * ratio, 0)
            x2 = min(bbox1[2] + w * ratio, width)
            y2 = min(bbox1[3] + h * ratio, height)
            bbox = np.array([x1, y1, x2, y2])
        else:
            bbox = np.array([0, 0, 0, 0])
        return bbox

    def initialize(self, image, mask):
        self.tracker = self.aot_tracker
        self.tracker.add_first_frame(image, mask)
        self.aot_mix_tracker = None
        self.mask_size = mask.shape

    def track(self, image):
        m, confidence = self.tracker.track(image)
        m = F.interpolate(torch.tensor(m)[None, None, :, :],
                          size=self.mask_size, mode="nearest").numpy().astype(np.uint8)[0][0]

        if self.sam_refine:
            obj_list = np.unique(m)
            mask_ = np.zeros_like(m)
            mask_2 = np.zeros_like(m)
            masks_ls = []
            for i in obj_list:
                mask = (m == i).astype(np.uint8)
                if i == 0 or mask.sum() == 0:
                    masks_ls.append(mask_)
                    continue
                bbox = self.get_box(mask)
                # box prompt
                self.mask_prompt.set_image(image)
                masks_, iou_predictions, _ = self.mask_prompt.predict(box=bbox)
                select_index = list(iou_predictions).index(max(iou_predictions))
                output = masks_[select_index].astype(np.uint8)
                iou = pytorch_iou(torch.from_numpy(output).cuda().unsqueeze(0),
                                  torch.from_numpy(mask).cuda().unsqueeze(0), [1])
                iou = iou.cpu().numpy()
                if iou < self.sam_refine_iou:
                    output = mask
                masks_ls.append(output)
                mask_2 = mask_2 + output * i
            masks_ls = np.stack(masks_ls)
            masks_ls_ = masks_ls.sum(0)
            masks_ls_argmax = np.argmax(masks_ls, axis=0)
            rs = np.where(masks_ls_ > 1, masks_ls_argmax, mask_2)
            rs = np.array(rs).astype(np.uint8)

            return rs, confidence
        return m, confidence


def mask_draw(msk):
    palette = np.reshape(palette_template, (-1, 1, 1, 3))
    masks = np.zeros_like(np.stack([msk[0], msk[0], msk[0]], -1)).astype(np.uint8)
    for i, j in enumerate(msk):
        msk_ = np.stack([j, j, j], -1) * (palette[i + 1])
        msk_ = np.stack([msk_[:, :, -1], msk_[:, :, 1], msk_[:, :, 0]], -1)
        msk_ = msk_.astype(np.uint8)
        masks = masks + msk_
    img = masks.astype(np.uint8)
    return img

############################### VOT running script
###############################################################################
set_Tracker = 'HQTrack'
vis_results = True
local_track = False
sam_refine = True
sam_refine_iou = 10
sam_refine_iou/=100.0
muti_object = True
save_mask = False
confidence_setto_1 = True

save_dir = os.path.join(os.path.dirname(__file__), '..', 'VOTS23_workspace/temp_vis')
mask_save_dir = os.path.join(os.path.dirname(__file__), '..', 'VOTS23_workspace/temp_mask')
epoch_num=42000

if set_Tracker == 'HQTrack':
    config = {
        'exp_name': 'default',
        'model': 'internT_msdeaotl_v2',
        'pretrain_model_path': 'result/default_InternT_MSDeAOTL_V2/YTB_DAV_VIP/ckpt/save_step_{}.pth'.format(epoch_num),
        'gpu_id': 0,
    }

# set cfg
if set_Tracker in ['HQTrack']:
    engine_config = importlib.import_module('configs.' + 'ytb_vip_dav_deaot_internT')
cfg = engine_config.EngineConfig(config['exp_name'], config['model'])
cfg.TEST_CKPT_PATH = os.path.join(AOT_PATH, config['pretrain_model_path'])

palette_template = Image.open(os.path.join(os.path.dirname(__file__), '..', 'my_tools/mask_palette.png')).getpalette()

# get first frame and mask
handle = vots_utils.VOT("mask",multiobject=True)
# selection = handle.region()
objects = handle.objects()
imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

# get first frame and mask
image_init = read_img(imagefile)
seq_name = imagefile.split('/')[-3]
# multi-start生成时间文件夹
if vis_results:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # cur_time = int(time.time() % 10000)
    save_dir = os.path.join(save_dir, seq_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

# mask = make_full_size(selection, (image.shape[1], image.shape[0]))
mask_objects_init = [(make_full_size(object, (image_init.shape[1], image_init.shape[0]))>0).astype(np.uint8) for object in objects]
object_nums = len(mask_objects_init)
# mask = (mask > 0).astype(np.uint8)


cfg.TEST_FLIP=False
cfg.TEST_MULTISCALE=[1.0]

if muti_object:
    tracker = MSAOTTracker(cfg, config, local_track, sam_refine,sam_refine_iou)

if muti_object:
    for i,mask in enumerate(mask_objects_init):
        if i==0:
            temp = mask
        else:
            temp += mask*(i+1)
    mask = temp
    tracker.initialize(image_init, mask)

init_flag = True
history_mask = None
while True:
    # count += 1
    imagefile = handle.frame()
    if not imagefile:
        break
    image = read_img(imagefile)

    pred_masks = []
    pred_confidences = []
    if muti_object:
        m, confidence = tracker.track(image)
        if confidence_setto_1:
            confidence = 1
        for i in range(len(mask_objects_init)):
            m_temp = m.copy()
            m_temp[m_temp!=i+1]=0
            m_temp[m_temp!=0]=1
            pred_masks.append(m_temp)
            pred_confidences.append(confidence)

    handle.report(pred_masks, pred_confidences)