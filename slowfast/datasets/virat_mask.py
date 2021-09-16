"""
@Author: Du Yunhao
@Filename: dataset.py
@Contact: dyh_bupt@163.com
@Time: 2021/8/11 11:19
@Discription: VIRAT Dataset
"""
import os
import copy
import json
import torch
import random
import numpy as np
from PIL import Image
from os.path import join, exists
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from .virat_utils import dict_class_id
from slowfast.utils import logging
from slowfast.datasets.build import DATASET_REGISTRY
from slowfast.datasets.random_erasing import RandomErasing
from slowfast.datasets.transform import create_random_augment
from slowfast.datasets.utils import get_random_sampling_rate, spatial_sampling,\
    tensor_normalize, pack_pathway_output, spatial_sampling_bbox

logger = logging.get_logger(__name__)

'''获取输入帧'''
def get_frames(clip_info, num_frames, clip_idx):
    """
    :param clip_info: clip信息
    :param num_frames: 采样帧数
    :param clip_idx: 采样模式，-1-随机采样，否则均匀采样
    :return: frames(tensor)
    """
    '''获取采样帧'''
    clip_len = clip_info['len']  # clip长度
    f_start = clip_info['fid']  # 起始帧
    f_stop = f_start + clip_len - 1  # 结束帧
    # f_sampled = sorted(random.sample(range(f_start, f_stop + 1), num_frames)) \
    #     if clip_idx == -1 else np.linspace(f_start, f_stop, num_frames, dtype=int).tolist()  # 采样帧
    f_sampled = np.linspace(f_start, f_stop, num_frames, dtype=int).tolist()
    '''读取图像'''
    dir_ = clip_info['dir']
    list_frames = list()  # list of [C,H,W]
    for fid in f_sampled:
        path = join(dir_, 'frame_%09d.jpg' % fid)
        if exists(path):
            list_frames.append(transforms.ToTensor()(Image.open(path)))
    assert len(list_frames) > 0
    num_missing = num_frames - len(list_frames)  # 考虑部分图像路径不存在，需进行补充
    list_frames += [list_frames[-1]] * num_missing
    '''裁剪图像'''
    bbox = clip_info['box'] # 事件大框
    person_object_bbox = clip_info['smallbox'] # 事件小框
    if isinstance(bbox[0], list):  # 逐帧框
        list_fid = list(range(f_start, f_stop + 1))
        width = np.mean([b[2] - b[0] for b in bbox], dtype=int)
        height = np.mean([b[3] - b[1] for b in bbox], dtype=int)
        for i, fid in enumerate(f_sampled):
            idx_box = list_fid.index(fid)
            x1, y1, x2, y2 = bbox[idx_box]  # 该帧的框
            croped = list_frames[i][:, y1:y2, x1:x2]  # 裁剪
            list_frames[i] = transforms.Resize((height, width))(croped)  # 放缩
    else:  # clip大框
        x1, y1, x2, y2 = list(map(int, bbox))
        list_frames = [f[:, y1:y2, x1:x2] for f in list_frames]  # 裁剪
        if person_object_bbox is not None:
            list_fid = list(range(f_start, f_stop + 1))
            # 初始化各种list
            person_masks = []
            object_masks = []
            list_person_mask = []
            list_object_mask = []
            # 将person与car的框分开，并将坐标统一到截图的坐标系中
            # person_mask = torch.zeros([1, y2-y1, x2-x1])
            # object_mask = torch.zeros([1, y2-y1, x2-x1])
            for ii,person_object in enumerate(person_object_bbox):
                # person_mask = torch.zeros([1, y2-y1, x2-x1])
                # object_mask = torch.zeros([1, y2-y1, x2-x1])
                person_mask = torch.zeros([y2-y1, x2-x1])
                object_mask = torch.zeros([y2-y1, x2-x1])
                for box in person_object:
                    if box[4] != 'person':
                        object_box, object_mask = Relocation_gaussian(object_mask,bbox,box[:4])
                    else:
                        person_box, person_mask = Relocation_gaussian(person_mask,bbox,box[:4])
                person_masks.append(person_mask.unsqueeze(0))
                object_masks.append(object_mask.unsqueeze(0))
                # person_masks.append(person_mask)
                # object_masks.append(object_mask)

            for i, fid in enumerate(f_sampled):
                idx_box = list_fid.index(fid)

                # 取mask
                list_person_mask.append(person_masks[idx_box])
                list_object_mask.append(object_masks[idx_box])

    #组合
    frames = torch.stack(list_frames)  # 拼接,[T,C,H,W]
    frames = frames.permute(0, 2, 3, 1)  # [T,C,H,W]→[T,H,W,C]
    if person_object_bbox is not None:
        person_masks = torch.stack(list_person_mask)    #[T,C,H,W]
        object_masks = torch.stack(list_object_mask)
        masks = torch.cat((person_masks,object_masks),dim=1) #[T,C,H,W] Channel#0为person， Channel#1为car
        # masks = torch.cat((person_mask,object_mask),dim=0) #[T,C,H,W] Channel#0为person， Channel#1为car
        # masks = masks.unsqueeze(0)
        masks = masks.permute(0,2,3,1)
        return frames, masks
    return frames, None

def Relocation(mask, clip_bbox, object_bbox):
    clip_x1,clip_y1,clip_x2,clip_y2 = clip_bbox
    object_x1,object_y1,object_x2,object_y2 = object_bbox
    object_x1 = max(0,int(object_x1 - clip_x1))
    object_y1 = max(0,int(object_y1 - clip_y1))
    object_x2 = max(0,int(object_x2 - clip_x1))
    object_y2 = max(0,int(object_y2 - clip_y1))
    relocation_bbox = [object_x1,object_y1,object_x2,object_y2]

    #generate position mask
    # mask = torch.zeros([1, clip_y2-clip_y1, clip_x2-clip_x1])
    mask[:,object_y1:object_y2,object_x1:object_x2] = 1
    return relocation_bbox,mask

def Relocation_gaussian(mask, clip_bbox, object_bbox):
    clip_x1,clip_y1,clip_x2,clip_y2 = clip_bbox
    object_x1,object_y1,object_x2,object_y2 = object_bbox
    object_x1 = max(0,int(object_x1 - clip_x1))
    object_y1 = max(0,int(object_y1 - clip_y1))
    object_x2 = max(0,int(object_x2 - clip_x1))
    object_y2 = max(0,int(object_y2 - clip_y1))

    cen = ((object_x1 + object_x2) / 2, (object_y1 + object_y2) / 2)
    w = object_x2 - object_x1
    h = object_y2 - object_y1
    mask = gen_gaussian_target(mask, cen, w/2, h/2)
    return None, mask

def gaussian2D(radius_x, radius_y, sigma_x=1, sigma_y=1, dtype=torch.float32, device='cpu'):
    """Generate 2D gaussian kernel.

    Args:
        radius (int): Radius of gaussian kernel.
        sigma (int): Sigma of gaussian function. Default: 1.
        dtype (torch.dtype): Dtype of gaussian tensor. Default: torch.float32.
        device (str): Device of gaussian tensor. Default: 'cpu'.

    Returns:
        h (Tensor): Gaussian kernel with a
            ``(2 * radius + 1) * (2 * radius + 1)`` shape.
    """
    x = torch.arange(
        -radius_x, radius_x + 1, dtype=dtype, device=device).view(1, -1)
    y = torch.arange(
        -radius_y, radius_y + 1, dtype=dtype, device=device).view(-1, 1)

    # h = (-(x * x + y * y) / (2 * sigma_x * sigma_y)).exp()
    h = (-((x * x / (2 * sigma_x * sigma_x)) + (y * y / (2 * sigma_y * sigma_y)))).exp()

    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h


def gen_gaussian_target(heatmap, center, radius_x, radius_y, k=1):
    """Generate 2D gaussian heatmap.

    Args:
        heatmap (Tensor): Input heatmap, the gaussian kernel will cover on
            it and maintain the max value.
        center (list[int]): Coord of gaussian kernel's center.
        radius (int): Radius of gaussian kernel.
        k (int): Coefficient of gaussian kernel. Default: 1.

    Returns:
        out_heatmap (Tensor): Updated heatmap covered by gaussian kernel.
    """
    radius_x = int(radius_x)
    radius_y = int(radius_y)
    diameter_x = 2 * radius_x + 1
    diameter_y = 2 * radius_y + 1

    gaussian_kernel = gaussian2D(
        radius_x, radius_y, sigma_x=diameter_x / 6, sigma_y=diameter_y / 6, dtype=heatmap.dtype, device=heatmap.device)

    x, y = center
    x = int(x)
    y = int(y)

    height, width = heatmap.shape[:2]

    left, right = min(x, radius_x), min(width - x, radius_x + 1)
    top, bottom = min(y, radius_y), min(height - y, radius_y + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian_kernel[radius_y - top:radius_y + bottom,
                                      radius_x - left:radius_x + right]
    out_heatmap = heatmap
    torch.max(
        masked_heatmap,
        masked_gaussian * k,
        out=out_heatmap[y - top:y + bottom, x - left:x + right])

    return out_heatmap

@DATASET_REGISTRY.register()
class Viratmask(Dataset):
    def __init__(
            self,
            cfg,
            mode,
            box_mode='clip',
            min_len=8,
            dict_class_id=dict_class_id
            # dict_class_id={'person_person_interaction': 13, 'person_talks_to_person': 23}
    ):
        """
        :param cfg: 配置信息；
        :param mode: 模式，'train','val','test'；
        :param num_retries: 读取尝试次数；
        :param box_mode: 目标框模式，'clip'-单个Clip大框，'frame'-逐帧小框；
        :param min_len: clip最小长度；
        :param dict_class_id: 类别与ID映射表；
        """
        assert mode in ['train', 'val', 'test']
        assert box_mode in ['clip', 'frame']
        # assert len(dict_class_id) == cfg.MODEL.NUM_CLASSES
        '''基本信息'''
        self.cfg = cfg
        self.mode = mode
        self._box_mode = box_mode
        self._min_len = min_len
        self._dict_class_id = dict_class_id
        '''
        train/val: 每个video仅采样单个clip;
        test: 每个视频采样NUM_ENSEMBLE_VIEWS个clip，每个clip空间裁剪NUM_SPATIAL_CROPS次;
        '''
        assert cfg.TEST.NUM_ENSEMBLE_VIEWS == 1, "当前仅支持Views=1"
        self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS if mode == 'test' else 1
        '''构建数据集'''
        logger.info("Constructing VIRAT {}...".format(mode))
        self._construct_loader()
        self.aug = False
        self.rand_erase = False
        if mode == 'train' and self.cfg.AUG.ENABLE:
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:
                self.rand_erase = True

    def _construct_loader(self):
        """
        构建数据集
        """
        name = 'val' if self.mode == 'test' else self.mode
        # path_to_file = join(self.cfg.DATA.PATH_TO_DATA_DIR, '{}_activities_clip.json'.format(name))
        # path_to_file = join(self.cfg.DATA.PATH_TO_DATA_DIR, '{}_clip_duration32_stride16.json'.format(name))
        if name == 'train':
            path_to_file = join(self.cfg.DATA.PATH_TO_DATA_DIR, '{}_clip_duration32_stride16_addbg_randomhalf.json'.format(name))
        else:
            path_to_file = join(self.cfg.DATA.PATH_TO_DATA_DIR, '{}_clip_duration32_stride16.json'.format(name))
        # path_to_file = join(self.cfg.DATA.PATH_TO_DATA_DIR, 'tmp.json')
        annotation = json.load(open(path_to_file, 'r'))
        self._clip_info = list()
        self._labels = list()
        self._spatial_temporal_idx = []
        '''遍历每个样本'''
        for dict_anno in annotation:
            # root_video = join(self.cfg.DATA.PATH_PREFIX, 'train' if self.mode == 'train' else 'val_test')
            root_video = self.cfg.DATA.PATH_PREFIX
            if dict_anno['duration'] < self._min_len: continue  # 滤除过短clip
            dict_anno_label = dict(filter(lambda kv: kv[0] in self._dict_class_id, dict_anno['label'].items()))  # 保留特定类
            if dict_anno_label:
                x1, y1, x2, y2 = dict_anno['bbox_clip']
                # assert x1 < dict_anno['bbox_prop'][0] and x2 > dict_anno['bbox_prop'][2] and y1 < dict_anno['bbox_prop'][1] and y2 > dict_anno['bbox_prop'][3]
                clip_info = {
                    'inverse': False,
                    'dir': join(root_video, dict_anno['video']),  # 帧前缀
                    'fid': dict_anno['fid'],  # 起始帧
                    'len': dict_anno['duration'],  # clip长度
                    'box': dict_anno['bbox_clip'] if self._box_mode == 'clip' \
                        else [x['big'] for x in dict_anno['bbox_frame'].values()],  # clip框或逐帧框
                    # 'box_object': list(map(lambda x :x[0]-x[1] ,zip(dict_anno['bbox_prop'],[x1, y1, x1, y1])))
                }  # 帧信息
                if name == 'train':
                    clip_info['smallbox'] = [x['small'] for x in dict_anno["bbox_frame"].values()]
                else:
                    clip_info['smallbox'] = None
                label = {self._dict_class_id[k]: dict_anno_label[k] if k in dict_anno_label else 0. for k in self._dict_class_id}  # 标签，例：{0:1.0, 1:0.2, ...}
                label = torch.tensor([label[k] for k in sorted(label)], dtype=torch.float64)
                '''遍历所有clip(考虑VIEWS和CROPS)'''
                for idx in range(self._num_clips):
                    self._clip_info.append(clip_info),
                    self._labels.append(label)
                    self._spatial_temporal_idx.append(idx)
                '''
                # 增加倒序样本
                if label[3] and self.mode == 'train':
                    clip_info_inverse = clip_info.copy()
                    clip_info_inverse['inverse'] = True
                    for idx in range(self._num_clips):
                        self._clip_info.append(clip_info_inverse),
                        self._labels.append(label)
                        self._spatial_temporal_idx.append(idx)
                '''
        assert len(self._labels) > 0
        logger.info(
            "Constructing VIRAT dataloader (size: {}) from {}".format(len(self._labels), path_to_file)
        )

    def __getitem__(self, index):
        """
        :return:
            frames (tensor): [C,T,H,W]
            label (dict): e.g., {1:1.0, 3:0.2}
            index (int): index
        """
        '''读取index'''
        short_cycle_idx = None
        if isinstance(index, tuple):
            index, short_cycle_idx = index
        '''裁剪采样参数'''
        if self.mode in ['train', 'val']:
            temporal_sample_index = -1  # -1: random sampling
            spatial_sample_index = -1   # -1: random sampling
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx] * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                min_scale = int(
                    round(
                        float(min_scale) * crop_size / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )  # Decreasing the scale is equivalent to using a larger "span" in a sampling grid.
        else:  # 'test'
            temporal_sample_index = self._spatial_temporal_idx[index] // self.cfg.TEST.NUM_SPATIAL_CROPS
            spatial_sample_index = self._spatial_temporal_idx[index] % self.cfg.TEST.NUM_SPATIAL_CROPS \
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1 else 1  # 0-left/top, 1-center/middle, 2-right/bottom
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3 \
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1 \
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2 + [self.cfg.DATA.TEST_CROP_SIZE]
        '''输入帧读取'''
        frames, masks = get_frames(
            clip_info=self._clip_info[index],
            num_frames=self.cfg.DATA.NUM_FRAMES,
            clip_idx=temporal_sample_index,
        )  # tensor [T,C,H,W]
        # if self._clip_info[index]['inverse']:
        #     frames = torch.flip(frames, dims=[0])
        '''输入帧处理'''
        if self.aug:
            if self.cfg.AUG.NUM_SAMPLE > 1:
                frame_list, label_list, index_list = list(), list(), list()
                for _ in range(self.cfg.AUG.NUM_SAMPLE):
                    new_frames = self._aug_frame(frames, spatial_sample_index, min_scale, max_scale, crop_size)
                    label = self._labels[index]
                    new_frames = pack_pathway_output(self.cfg, new_frames)
                    frame_list.append(new_frames)
                    label_list.append(label)
                    index_list.append(index)
                return frame_list, label_list, index_list, {}
            else:
                frames = self._aug_frame(frames, spatial_sample_index, min_scale, max_scale, crop_size)
        else:
            frames = tensor_normalize(frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD)
            frames = frames.permute(3, 0, 1, 2)  # [T,H,W,C]→[C,T,H,W]
            if masks is not None:
                masks = masks.permute(3, 0, 1, 2)

                # boxes = np.array([self._clip_info[index]['box_object']])
                frames, masks = spatial_sampling_mask(
                    frames,
                    masks,
                    # boxes,
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                    random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                    inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                )
            else:
                frames = spatial_sampling(
                    frames,
                    # boxes,
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                    random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                    inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                )
        label = self._labels[index]
        
        frames = pack_pathway_output(self.cfg, frames)
        reg_label = {}
        if self.mode == 'train':
            masks = masks.resize_(masks.shape[0], masks.shape[1], masks.shape[2]//4, masks.shape[3]//4)
            masks = pack_pathway_output(self.cfg, masks)
            return frames, {}, label, {}, index, {}, reg_label, masks
        else:
            return frames, label, index, {}

    def _aug_frame(self, frames, spatial_sample_index, min_scale, max_scale, crop_size,):
        aug_transform = create_random_augment(
            input_size=(frames.size(1), frames.size(2)),
            auto_augment=self.cfg.AUG.AA_TYPE,
            interpolation=self.cfg.AUG.INTERPOLATION,
        )
        frames = frames.permute(0, 3, 1, 2)  # [T,H,W,C]→[T,C,H,W]
        list_img = [transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))]
        list_img = aug_transform(list_img)
        frames = torch.stack([transforms.ToTensor()(img) for img in list_img])
        frames = frames.permute(0, 2, 3, 1)  # [T,C,H,W]→[T,H,W,C]
        frames = tensor_normalize(frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD)
        frames = frames.permute(3, 0, 1, 2)  # [T,H,W,C]→[C,T,H,W]
        # Perform data augmentation.
        scl, asp = (
            self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE,
            self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
        )
        relative_scales = (
            None if (self.mode not in ["train"] or len(scl) == 0) else scl
        )
        relative_aspect = (
            None if (self.mode not in ["train"] or len(asp) == 0) else asp
        )
        frames = spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            aspect_ratio=relative_aspect,
            scale=relative_scales,
            motion_shift=self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT if self.mode in ["train"] else False
        )
        if self.rand_erase:
            erase_transform = RandomErasing(
                self.cfg.AUG.RE_PROB,
                mode=self.cfg.AUG.RE_MODE,
                max_count=self.cfg.AUG.RE_COUNT,
                num_splits=self.cfg.AUG.RE_COUNT,
                device="cpu",
            )
            frames = frames.permute(1, 0, 2, 3)
            frames = erase_transform(frames)
            frames = frames.permute(1, 0, 2, 3)
        return frames

    def __len__(self):
        return self.num_videos

    @property
    def num_videos(self):
        return len(self._labels)

def spatial_sampling_mask(
    frames, masks,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    from . import transform as transform
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        assert aspect_ratio is None
        assert scale is None
        if aspect_ratio is None and scale is None:
            frames, _ , masks= random_short_side_scale_jitter_gcn(
                images=frames,
                masks=masks,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, masks = random_crop_gcn(frames, masks, crop_size)
        else:
            # not supported yet
            transform_func = (
                transform.random_resized_crop_with_shift
                if motion_shift
                else transform.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _, masks = horizontal_flip_gcn(0.5, frames, masks)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale}) == 1
        frames, _ = transform.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)

    return frames, masks

def horizontal_flip_gcn(prob, images, masks,boxes=None):
    if boxes is None:
        flipped_boxes = None
    else:
        flipped_boxes = boxes.copy()

    if np.random.uniform() < prob:
        images = images.flip((-1))
        masks =  masks.flip((-1))

        if len(images.shape) == 3:
            width = images.shape[2]
        elif len(images.shape) == 4:
            width = images.shape[3]
        else:
            raise NotImplementedError("Dimension does not supported")
        if boxes is not None:
            flipped_boxes[:, [0, 2]] = width - boxes[:, [2, 0]] - 1

    return images, flipped_boxes, masks

def random_short_side_scale_jitter_gcn(images, masks, min_size, max_size, boxes=None, inverse_uniform_sampling=False):
    import math
    if inverse_uniform_sampling:
        size = int(
            round(1.0 / np.random.uniform(1.0 / max_size, 1.0 / min_size))
        )
    else:
        size = int(round(np.random.uniform(min_size, max_size)))

    height = images.shape[2]
    width = images.shape[3]
    if (width <= height and width == size) or (
        height <= width and height == size
    ):
        return images, boxes, masks
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
        if boxes is not None:
            boxes = boxes * float(new_height) / height
    else:
        new_width = int(math.floor((float(width) / height) * size))
        if boxes is not None:
            boxes = boxes * float(new_width) / width

    return (
        torch.nn.functional.interpolate(
            images,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        ),
        boxes,
        torch.nn.functional.interpolate(
            masks,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        ),
    )

def uniform_crop_gcn(images, masks, size, spatial_idx, boxes=None, scale_size=None):
    import math
    assert spatial_idx in [0, 1, 2]
    ndim = len(images.shape)
    if ndim == 3:
        images = images.unsqueeze(0)
    height = images.shape[2]
    width = images.shape[3]
    if scale_size is not None:
        if width <= height:
            width, height = scale_size, int(height / width * scale_size)
        else:
            width, height = int(width / height * scale_size), scale_size
        images = torch.nn.functional.interpolate(
            images,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))
    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size
    cropped = images[:, :, y_offset : y_offset + size, x_offset : x_offset + size]
    cropped_mask = masks[:, :, y_offset : y_offset + size, x_offset : x_offset + size]
    
    new_person_bbox = []
    for person_box in person_bbox:
        x1,y1,x2,y2 = person_box
        x1 = x1 - x_offset
        y1 = y1 - x_offset
        x2 = x2 - x_offset
        y2 = y2 - x_offset
        person_box = [x1,y1,x2,y2]
        new_person_bbox.append(person_box)

    new_car_bbox = []
    for car_box in car_bbox:
        x1,y1,x2,y2 = car_box
        x1 = x1 - x_offset
        y1 = y1 - x_offset
        x2 = x2 - x_offset
        y2 = y2 - x_offset
        car_box = [x1,y1,x2,y2]
        new_car_bbox.append(car_box)

    if ndim == 3:
        cropped = cropped.squeeze(0)
    return cropped, cropped_mask, new_person_bbox, new_car_bbox

def random_crop_gcn(images, masks, size):
    if images.shape[2] == size and images.shape[3] == size:
        return images
    height = images.shape[2]
    width = images.shape[3]
    y_offset = 0
    if height > size:
        y_offset = int(np.random.randint(0, height - size))
    x_offset = 0
    if width > size:
        x_offset = int(np.random.randint(0, width - size))
    cropped = images[:, :, y_offset : y_offset + size, x_offset : x_offset + size]
    cropped_masks = masks[:,:,y_offset : y_offset + size, x_offset : x_offset + size]

    return cropped, cropped_masks
if __name__ == '__main__':
    from slowfast.utils.parser import load_config, parse_args
    from slowfast.config.defaults import assert_and_infer_cfg
    args = parse_args()
    args.cfg_file = '/home/sugar/workspace/slowfast/configs/Virat/X3D_M.yaml'
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    dataset = Virat(
        cfg=cfg,
        mode='train',
        box_mode='clip',
        min_len=8,
        dict_class_id=dict_class_id,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
    )
    for i, (frames, label, idx, _) in enumerate(dataloader):
        print('{}/{}'.format(i + 1, len(dataset)), frames[0].size(), label, idx)