import os
import glob
import torch
import cv2
import numpy as np
import random
import sys
import torch.nn.functional as F
from PIL import Image
import importlib
from torchvision import transforms
from os.path import join
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.image import flip_tensor
AOT_PATH = os.path.join(os.path.dirname(__file__), '..')
import dataloaders.video_transforms as tr

from networks.engines import build_engine
from utils.checkpoint import load_network
from networks.models import build_vos_model
from utils.metric import pytorch_iou

base_path = os.path.dirname(os.path.abspath(__file__))
# video for test
demo_video = 'bolt'
img_files = sorted(glob.glob(join(base_path, demo_video, '*.jp*')))
point_box_prompts=[]

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

seed_torch(1000000007)
torch.set_num_threads(4)
torch.autograd.set_grad_enabled(False)

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
        self.model.cuda(gpu_id)
        self.model.eval()
        print('cfg.TEST_CKPT_PATH = ', cfg.TEST_CKPT_PATH)
        self.model, _ = load_network(self.model, cfg.TEST_CKPT_PATH, gpu_id)
        self.aug_nums = len(cfg.TEST_MULTISCALE)
        if cfg.TEST_FLIP:
            self.aug_nums *= 2
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
            'height': frame.shape[0],
            'weight': frame.shape[1]
        }
        sample = self.transform(sample)

        if self.aug_nums > 1:
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

        if self.aug_nums > 1:
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
            pred_prob = torch.mean(cat_all_preds, dim=0, keepdim=True)
            pred_label = torch.argmax(pred_prob, dim=1, keepdim=True).float()
            _pred_label = F.interpolate(pred_label,
                                        size=self.engine[aug_idx].input_size_2d,
                                        mode="nearest")
            self.engine[aug_idx].update_memory(_pred_label)
            mask = pred_label.detach().cpu().numpy()[0][0].astype(np.uint8)
        conf = 0

        return mask, conf

def read_img(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class HQTrack(object):
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

def OnMouse_box(event,x,y,flags,param):
    global x0, y0, img4show, img
    if event == cv2.EVENT_LBUTTONDOWN:
        x0,y0 =x,y
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        x_temp, y_temp = x, y
        img4show=img.copy()
        cv2.rectangle(img4show, (x0, y0), (x_temp, y_temp), (255, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        x1, y1 = x, y
        cv2.rectangle(img4show, (x0, y0), (x, y), (255, 255, 0), 2)
        img=img4show
        point_box_prompts.append([x0, y0, x1, y1])

def OnMouse_point(event,x,y,flags,param):
    global x0, y0, img4show, img
    if event == cv2.EVENT_LBUTTONDOWN:
        x0,y0 =x,y
        # print(x0,y0)
        point_box_prompts.append([x0,y0])
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.circle(img4show, (x0, y0), 4, (0, 255, 0), 6)
        img=img4show


# SAM
print("SAM init ...")
model_type = 'vit_l'
sam_checkpoint = os.path.join(base_path, '..', 'segment_anything_hq/pretrained_model/sam_hq_vit_l.pth')
output_mode = "binary_mask"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=torch.device('cuda'))
mask_generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode)
mask_prompt = SamPredictor(sam)

# HQTrack config
# choose point or box prompt for SAM
SAM_prompt = 'Point' #'Box
set_Tracker = 'HQTrack'
sam_refine = True
sam_refine_iou = 0.1
muti_object = True
epoch_num=42000
config = {
        'exp_name': 'default',
        'model': 'internT_msdeaotl_v2',
        'pretrain_model_path': 'result/default_InternT_MSDeAOTL_V2/YTB_DAV_VIP/ckpt/save_step_{}.pth'.format(epoch_num),
        'gpu_id': 0,}
# set cfg
print('VMOS init ...')
if set_Tracker in ['HQTrack']:
    engine_config = importlib.import_module('configs.' + 'ytb_vip_dav_deaot_internT')
cfg = engine_config.EngineConfig(config['exp_name'], config['model'])
cfg.TEST_CKPT_PATH = os.path.join(AOT_PATH, config['pretrain_model_path'])
palette_template = Image.open(os.path.join(os.path.dirname(__file__), '..', 'my_tools/mask_palette.png')).getpalette()
tracker = HQTrack(cfg, config, True, sam_refine,sam_refine_iou)
save_dir = './output'

for idx,img_file in enumerate(img_files):
    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_ori=img.copy()
    # Select ROI
    if idx == 0:
        img4show = img.copy()
        while (1):
            cv2.namedWindow("demo")
            cv2.imshow('demo', cv2.cvtColor(img4show, cv2.COLOR_RGB2BGR))
            if SAM_prompt == 'Box':
                OnMouse = OnMouse_box
            elif SAM_prompt == 'Point':
                OnMouse = OnMouse_point

            cv2.setMouseCallback('demo', OnMouse)
            k = cv2.waitKey(1)
            if k == ord('r'):
                break
        # point prompt
        masks_ls = []
        mask_2 = np.zeros_like(img[:,:,0])
        masks_ls.append(mask_2)
        for obj_idx, prompt in enumerate(point_box_prompts):
            mask_prompt.set_image(img_ori)
            if SAM_prompt == 'Box':
                masks_, iou_predictions, _ = mask_prompt.predict(box=np.array(prompt).astype(float))
            elif SAM_prompt == 'Point':
                masks_, iou_predictions, _ = mask_prompt.predict(point_labels=np.asarray([1]), point_coords=np.asarray([prompt]))
            select_index = list(iou_predictions).index(max(iou_predictions))
            init_mask = masks_[select_index].astype(np.uint8)
            masks_ls.append(init_mask)
            mask_2 = mask_2 + init_mask * (obj_idx+1)
        masks_ls = np.stack(masks_ls)
        masks_ls_ = masks_ls.sum(0)
        masks_ls_argmax = np.argmax(masks_ls, axis=0)
        rs = np.where(masks_ls_ > 1, masks_ls_argmax, mask_2)
        rs = np.array(rs).astype(np.uint8)
        init_masks = []
        for i in range(len(masks_ls)):
            m_temp = rs.copy()
            m_temp[m_temp!=i+1]=0
            m_temp[m_temp!=0]=1
            init_masks.append(m_temp)
        # img+mask for vis
        img = cv2.cvtColor(img_ori.astype(np.float32), cv2.COLOR_RGB2BGR)
        for idx, m in enumerate(init_masks):
            img[:, :, 1] += 127.0 * m
            img[:, :, 2] += 127.0 * m
            contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            im_m = cv2.drawContours(img, contours, -1, cur_colors[idx], 2)
        im_m = im_m.clip(0, 255).astype(np.uint8)
        cv2.putText(im_m, 'Init', (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 5)
        cv2.imshow('demo', im_m)
        k = cv2.waitKey(1)
        # HQtrack init
        print('init target objects ...')
        tracker.initialize(img_ori, rs)
        obj_num = len(init_masks)
        print('HQTrack runing ...')
    else:
        m, confidence = tracker.track(img_ori)
        print('Running frame ', idx)
        pred_masks = []
        for i in range(obj_num):
            m_temp = m.copy()
            m_temp[m_temp != i + 1] = 0
            m_temp[m_temp != 0] = 1
            pred_masks.append(m_temp)
        img = cv2.cvtColor(img_ori.astype(np.float32), cv2.COLOR_RGB2BGR)
        for idx, m in enumerate(pred_masks):
            img[:, :, 1] += 127.0 * m
            img[:, :, 2] += 127.0 * m
            contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            im_m = cv2.drawContours(img, contours, -1, cur_colors[idx], 2)
        im_m = im_m.clip(0, 255).astype(np.uint8)
        save_path = os.path.join(save_dir, img_file.split('/')[-1])
        cv2.imwrite(save_path, im_m)
