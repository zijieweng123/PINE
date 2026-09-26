from PIL import Image, ImageOps
from os import path as osp
import torch
import torchvision.transforms.functional as TF
import cv2
from torch.utils import data as data
from pyiqa.utils.registry import DATASET_REGISTRY
from .base_iqa_dataset import BaseIQADataset
# from .preprocessing import *
import numpy as np
import random
import copy
# # 设置随机种子

def set_seed(seed=42):
    # worker_seed = torch.initial_seed() % 2**32 
    random.seed(seed)

    np.random.seed(seed)       
    torch.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
 
# set_seed(seed=114514)
set_seed(seed=215)

def get_spatial_fragments(
    video,
    fragments_h=7,
    fragments_w=7,
    fsize_h=32,
    fsize_w=32,
    aligned=32,
    nfrags=1,
    random=False,
    random_upsample=False,
    fallback_type="upsample",
    **kwargs,
):
    size_h = fragments_h * fsize_h
    size_w = fragments_w * fsize_w
    ## video: [C,T,H,W]
    ## situation for images
    if video.shape[1] == 1:
        aligned = 1

    dur_t, res_h, res_w = video.shape[-3:]
    ratio = min(res_h / size_h, res_w / size_w)
    if fallback_type == "upsample" and ratio < 1:
        
        ovideo = video
        video = torch.nn.functional.interpolate(
            video / 255.0, scale_factor=1 / ratio, mode="bilinear"
        )
        video = (video * 255.0).type_as(ovideo)
        
    if random_upsample:

        randratio = random.random() * 0.5 + 1
        video = torch.nn.functional.interpolate(
            video / 255.0, scale_factor=randratio, mode="bilinear"
        )
        video = (video * 255.0).type_as(ovideo)



    assert dur_t % aligned == 0, "Please provide match vclip and align index"
    size = size_h, size_w

    ## make sure that sampling will not run out of the picture
    hgrids = torch.LongTensor(
        [min(res_h // fragments_h * i, res_h - fsize_h) for i in range(fragments_h)]
    )
    wgrids = torch.LongTensor(
        [min(res_w // fragments_w * i, res_w - fsize_w) for i in range(fragments_w)]
    )
    hlength, wlength = res_h // fragments_h, res_w // fragments_w

    if random:
        print("This part is deprecated. Please remind that.")
        if res_h > fsize_h:
            rnd_h = torch.randint(
                res_h - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        if res_w > fsize_w:
            rnd_w = torch.randint(
                res_w - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
    else:
        if hlength > fsize_h:
            rnd_h = torch.randint(
                hlength - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        if wlength > fsize_w:
            rnd_w = torch.randint(
                wlength - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()

    target_video = torch.zeros(video.shape[:-2] + size).to(video.device)
    # target_videos = []

    for i, hs in enumerate(hgrids):
        for j, ws in enumerate(wgrids):
            for t in range(dur_t // aligned):
                t_s, t_e = t * aligned, (t + 1) * aligned
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w
                if random:
                    h_so, h_eo = rnd_h[i][j][t], rnd_h[i][j][t] + fsize_h
                    w_so, w_eo = rnd_w[i][j][t], rnd_w[i][j][t] + fsize_w
                else:
                    h_so, h_eo = hs + rnd_h[i][j][t], hs + rnd_h[i][j][t] + fsize_h
                    w_so, w_eo = ws + rnd_w[i][j][t], ws + rnd_w[i][j][t] + fsize_w
                target_video[:, t_s:t_e, h_s:h_e, w_s:w_e] = video[
                    :, t_s:t_e, h_so:h_eo, w_so:w_eo
                ]
    # target_videos.append(video[:,t_s:t_e,h_so:h_eo,w_so:w_eo])
    # target_video = torch.stack(target_videos, 0).reshape((dur_t // aligned, fragments, fragments,) + target_videos[0].shape).permute(3,0,4,1,5,2,6)
    # target_video = target_video.reshape((-1, dur_t,) + size) ## Splicing Fragments
    return target_video

# def get_salient_patch(image_path, block_size=384, show_saliency=False):
#     image = cv2.imread(image_path)
#     h, w = image.shape[:2]

#     # saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
#     saliency = cv2.saliency.StaticSaliencyFineGrained_create()
#     (success, saliencyMap) = saliency.computeSaliency(image)

#     saliencyMap = (saliencyMap * 255).astype("uint8")

#     if show_saliency:
#         cv2.imshow("Saliency Map", saliencyMap)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()


#     max_score = -1
#     best_patch = None
#     stride = block_size // 6 
#     for y in range(0, h - block_size + 1, stride):
#         for x in range(0, w - block_size + 1, stride):
#             patch_saliency = saliencyMap[y:y+block_size, x:x+block_size]
#             score = np.sum(patch_saliency)  
#             if score > max_score:
#                 max_score = score
#                 best_patch = image[y:y+block_size, x:x+block_size]

#     return best_patch

def five_crop_patches(img_pil, patch_size, offset=384):
    w, h = img_pil.size
    crops = [
        (offset, offset),  # top-left
        (w - patch_size - offset, offset),  # top-right
        (offset, h - patch_size - offset),  # bottom-left
        (w - patch_size - offset, h - patch_size - offset),  # bottom-right

        # 中心
        ((w - patch_size) // 2, (h - patch_size) // 2)  # center
    ]

    return [img_pil.crop((x, y, x + patch_size, y + patch_size)) for (x, y) in crops]

def map_mos_to_class(mos_label):

    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    class_id = sum(mos_label >= b for b in bins)  # 得到1~5
    class_id = max(1, min(class_id, 5))
    return class_id - 1


@DATASET_REGISTRY.register()
class HNRDataset(BaseIQADataset):
    """General No Reference dataset with meta info file.
    """
    def init_path_mos(self, opt):
        super().init_path_mos(opt)

        target_img_folder = opt['dataroot_target']

        self.paths_mos = []
        for row in self.meta_info.values:
            img_path = osp.join(target_img_folder, row[0])
            mos_label = float(row[1])
            self.paths_mos.append([img_path, mos_label])

    @staticmethod
    def _random_patch(img_pil, patch_size):
        """Return one random square patch, upsampling only if necessary."""
        if min(img_pil.size) < patch_size:
            img_pil = TF.resize(img_pil, patch_size)

        width, height = img_pil.size
        left = random.randint(0, width - patch_size)
        top = random.randint(0, height - patch_size)
        return img_pil.crop((left, top, left + patch_size, top + patch_size))

    def _random_patch_preprocess(self, img_pil, augment_opt):
        """Create train/eval inputs for the 25-random-patch strategy.

        Training uses one random patch per image. Validation uses the requested
        number of random patches; HRIQAModel averages their predictions.
        """
        patch_size = int(augment_opt.get('random_patch_size', 384))
        patch_count = int(augment_opt.get('random_patch_count', 25))

        if self.phase == 'train' and augment_opt.get('hflip', False) and random.random() < 0.5:
            img_pil = ImageOps.mirror(img_pil)

        count = 1 if self.phase == 'train' else patch_count
        patches = [self._random_patch(img_pil, patch_size) for _ in range(count)]
        tensors = torch.stack([TF.to_tensor(patch) for patch in patches], dim=0)
        return tensors[0] if count == 1 else tensors

    # def set_epoch(self, epoch):
    #     self.epoch = epoch
        
    def __getitem__(self, index):

        img_path = self.paths_mos[index][0]
        mos_label = float(self.paths_mos[index][1])
        cls_label = map_mos_to_class(mos_label)
        img_pil = Image.open(img_path).convert('RGB')
        # img_pil = dtcwt(img_path, J=3)
        
        # img_aesthetics = self.trans(img_pil)
        augment_opt = self.opt.get('augment', {})
        fragment_extent = augment_opt.get('salient_patch_dimension', 384)
        if isinstance(fragment_extent, (list, tuple)):
            h, w = fragment_extent
        else:
            h = w = fragment_extent
        
        preprocess_strategy = augment_opt.get('preprocess_strategy', 'fragment')

        if preprocess_strategy == 'random25_patch384':
            img_distortion = self._random_patch_preprocess(img_pil, augment_opt)
        elif preprocess_strategy in {
                'resize384',
                'resize_shortest_random_crop384',
        }:
            img_distortion = self.trans(img_pil)
        else:
            img_distortion = self.trans(img_pil)

        # The resize/crop strategies already produce the final network input.
        # Fragment mode below preserves the original HNR preprocessing path.
        if preprocess_strategy in {
                'resize384',
                'resize_shortest_random_crop384',
                'random25_patch384',
        }:
            mos_label_tensor = torch.Tensor([mos_label])
            cls_label_tensor = torch.tensor(cls_label, dtype=torch.long)
            return {
                'img_distortion': img_distortion,
                'mos_label': mos_label_tensor,
                'cls_label': cls_label_tensor,
                'img_path': img_path,
            }

        grid_size = augment_opt.get('fragment_grid_size')
        fragment_size = augment_opt.get('fragment_size', 32)
        if grid_size is None:
            f_h = int(h) // int(fragment_size)
            f_w = int(w) // int(fragment_size)
        elif isinstance(grid_size, (list, tuple)):
            f_h, f_w = int(grid_size[0]), int(grid_size[1])
        else:
            f_h = f_w = int(grid_size)
        fragment_size = int(fragment_size)
        expected_h = f_h * fragment_size
        expected_w = f_w * fragment_size
        if expected_h != int(h) or expected_w != int(w):
            raise ValueError(
                f'fragment_grid_size * fragment_size must match salient_patch_dimension, '
                f'got ({f_h}, {f_w}) * {fragment_size} = ({expected_h}, {expected_w}) '
                f'but salient_patch_dimension is ({int(h)}, {int(w)}).'
            )
        # print(f_h)
        img_distortion = img_distortion.unsqueeze(1)
        img_distortion = get_spatial_fragments(
            img_distortion,
            fragments_h=f_h,
            fragments_w=f_w,
            fsize_h=fragment_size,
            fsize_w=fragment_size,
            aligned=32,
            nfrags=1,
            random=False,
            random_upsample=False,
            fallback_type="upsample"
        )
        img_distortion = img_distortion.squeeze(1)
        # img_distortion = self.transform_distortion(img_distortion) 
        
        # sal_patches = five_crop_patches(img_pil, 384)
        # chosen_sal_patch = random.choice(sal_patches)
        # patch_idx = (index + self.epoch) % 5  # 每轮偏移选择不同 patch
        # chosen_sal_patch = sal_patches[patch_idx]
        # if self.phase != 'train':
        #     img_saliency = torch.stack([self.transform_totensor(p) for p in sal_patches])  # shape: [5, C, H, W]
        #     img_saliency = self.tansform_saliency(img_pil) 
        # else:
        #     # img_saliency = self.transform_totensor(chosen_sal_patch)
        #     img_saliency = self.transform_saliency(img_pil) 
        # img_saliency = self.transform_saliency(img_pil) 
        # img_saliency = get_salient_patch(img_path, 384)
        
        # img_dtcwt = self.transform_dtcwt(img_pil)

        mos_label_tensor = torch.Tensor([mos_label])
        cls_label_tensor = torch.tensor(cls_label, dtype=torch.long)
        # print("Unique class labels:", np.unique(cls_label_tensor))
        # print(cls_label_tensor.shape)
        return {
                # 'img_aesthetics': img_aesthetics, 
                'img_distortion': img_distortion, 
                # 'img_saliency': img_saliency,
                # 'img_dtcwt': img_dtcwt,
                'mos_label': mos_label_tensor, 
                'cls_label': cls_label_tensor,
                'img_path': img_path}
