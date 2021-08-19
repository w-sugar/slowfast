"""
@Author: Du Yunhao
@Filename: dataset.py
@Contact: dyh_bupt@163.com
@Time: 2021/8/11 11:19
@Discription: VIRAT Dataset
"""
import os
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
    tensor_normalize, pack_pathway_output

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
    bbox = clip_info['box']
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
        x1, y1, x2, y2 = bbox
        list_frames = [f[:, y1:y2, x1:x2] for f in list_frames]  # 裁剪
    frames = torch.stack(list_frames)  # 拼接,[T,C,H,W]
    frames = frames.permute(0, 2, 3, 1)  # [T,C,H,W]→[T,H,W,C]
    return frames

@DATASET_REGISTRY.register()
class Virat(Dataset):
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
        assert len(dict_class_id) == cfg.MODEL.NUM_CLASSES
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
                clip_info = {
                    'dir': join(root_video, dict_anno['video']),  # 帧前缀
                    'fid': dict_anno['fid'],  # 起始帧
                    'len': dict_anno['duration'],  # clip长度
                    'box': dict_anno['bbox_clip'] if self._box_mode == 'clip' \
                        else [x['big'] for x in dict_anno['bbox_frame'].values()],  # clip框或逐帧框
                }  # 帧信息
                label = {self._dict_class_id[k]: dict_anno_label[k] if k in dict_anno_label else 0. for k in self._dict_class_id}  # 标签，例：{0:1.0, 1:0.2, ...}
                label = torch.tensor([label[k] for k in sorted(label)], dtype=torch.float64)
                '''遍历所有clip(考虑VIEWS和CROPS)'''
                for idx in range(self._num_clips):
                    self._clip_info.append(clip_info),
                    self._labels.append(label)
                    self._spatial_temporal_idx.append(idx)
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
        frames = get_frames(
            clip_info=self._clip_info[index],
            num_frames=self.cfg.DATA.NUM_FRAMES,
            clip_idx=temporal_sample_index,
        )  # tensor [T,C,H,W]
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
            frames = spatial_sampling(
                frames,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            )
        label = self._labels[index]
        frames = pack_pathway_output(self.cfg, frames)
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

if __name__ == '__main__':
    from slowfast.utils.parser import load_config, parse_args
    from slowfast.config.defaults import assert_and_infer_cfg
    args = parse_args()
    args.cfg_file = '/home/dyh/CV_project/SlowFast/dyh/MVIT_B_16x4_CONV.yaml'
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    dataset = Virat(
        cfg=cfg,
        mode='val',
        box_mode='clip',
        min_len=8,
        dict_class_id={
            'person_sits': 20,
            'person_stands': 21,
        }
    )
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
    )
    for i, (frames, label, idx, _) in enumerate(dataloader):
        print('{}/{}'.format(i + 1, len(dataset)), frames[0].size(), label.size(), idx)