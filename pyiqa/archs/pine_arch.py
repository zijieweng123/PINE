"""PINE architecture for no-reference image quality assessment."""

import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from scipy.fftpack import dct

from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.archs.arch_util import load_pretrained_network, uniform_crop

import copy
from .topiq_swin import create_swin


class DownSampleFilter(nn.Module):

    def __init__(self):
        super().__init__()
        layer = nn.Conv2d(1, 1, kernel_size=5, stride=2, padding=2)
        kernel = np.float32([1, 4, 6, 4, 1])
        kernel = np.outer(kernel, kernel)
        weight = torch.from_numpy((kernel / kernel.sum()).reshape(1, 1, 5, 5))
        layer.weight = nn.Parameter(weight, requires_grad=False)
        layer.bias.data.zero_()
        layer.bias.requires_grad = False
        self.filter = layer

    def forward(self, x):
        if x.shape[1] == 1:
            return self.filter(x)
        batch_size, channels, height, width = x.shape
        output = self.filter(x.reshape(batch_size * channels, 1, height, width))
        return output.reshape(batch_size, channels, output.shape[2], output.shape[3])


class ConvRelu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel, padding="same", use_bias=True, dilation_rate=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=1,
            padding=padding,
            bias=use_bias,
            dilation=dilation_rate,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class DenseBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_relu_1 = ConvRelu(64, 64, kernel=3, dilation_rate=2)
        self.conv_relu_2 = ConvRelu(128, 192, kernel=3, dilation_rate=2)
        self.conv_relu_3 = ConvRelu(320, 64, kernel=3, dilation_rate=2)
        self.dense_block_num = 3

    def forward(self, x):
        first = self.conv_relu_1(x)
        first_concat = torch.cat([x, first], dim=1)
        second = self.conv_relu_2(first_concat)
        second_concat = torch.cat([first_concat, second], dim=1)
        third = self.conv_relu_3(second_concat)
        return torch.cat([second_concat, third], dim=1)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(self, src):
        src2 = self.norm1(src)
        q = k = src2

        src2, self.attn_map = self.self_attn(q, k, value=src2)

        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(self, tgt, memory):
        memory = self.norm2(memory)
        tgt2 = self.norm1(tgt)

        tgt2, self.attn_map = self.multihead_attn(query=tgt2, key=memory, value=memory)

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory)

        return output


class GatedConv(nn.Module):

    def __init__(self, weightdim, ksz=3):
        super().__init__()

        self.splitconv = nn.Conv2d(weightdim, weightdim * 2, 1, 1, 0)
        self.act = nn.GELU()

        self.weight_blk = nn.Sequential(
            nn.Conv2d(weightdim, 64, 1, stride=1), nn.GELU(), nn.Conv2d(64, 64, ksz, stride=1, padding=1), nn.GELU(),
            nn.Conv2d(64, 1, ksz, stride=1, padding=1), nn.Sigmoid())

    def forward(self, x):
        x1, x2 = self.splitconv(x).chunk(2, dim=1)
        weight = self.weight_blk(x2)
        x1 = self.act(x1)
        return x1 * weight


class CSFWeightGenerator_BT(nn.Module):
    """
    DCT域CSF权重生成器（基于Mannos-Sakrison模型）
    参考文献：
    [1] Mannos, J. L., & Sakrison, D. J. (1974).
        "The effects of a visual fidelity criterion on the encoding of images".
        IEEE Transactions on Information Theory, 20(4), 525-536.
    [2] Barten, P. G. (1999).
        "Contrast sensitivity of the human eye and its effects on image quality".
        SPIE Press.
    """

    def __init__(self, block_size, ppd=30.0):
        """
        Args:
            block_size: DCT分块大小（建议8/12/16）
            ppd: pixels per degree，每度视角像素数
                  典型值：
                  - 50cm观看距离+96PPI屏幕 ≈ 30
                  - 计算公式：ppd = (π/180)*观看距离*PPI
        """
        super().__init__()
        self.block_size = block_size
        self.ppd = ppd
        self.weights = self._generate_weights()
        self.csf_weight = self.weights

    def _generate_weights(self):
        """生成与DCT自然顺序对齐的Mannos-Sakrison CSF权重"""

        u, v = np.mgrid[0:self.block_size, 0:self.block_size].astype(np.float32)
        f_pixel = np.sqrt(u**2 + v**2) / (2.0 * self.block_size)

        f = f_pixel * self.ppd

        k = 1.0
        N0 = 0.0023
        fc = 3.0
        fe = 20.0

        with np.errstate(divide='ignore', invalid='ignore'):
            f_peak = 8.0
            max_val = 2.6 * (0.0192 + 0.114 * f_peak) *\
              math.exp(-(0.114 * f_peak) ** 1.1)

            csf = 2.6 * (0.0192 + 0.114 * f) * np.exp(-(0.114 * f)**1.1)

            csf[f > f_peak] = max_val

            csf[0, 0] = 0

        csf = csf / csf.max()
        return torch.from_numpy(csf).float()

    def forward(self, x_dct):
        """应用CSF权重到DCT系数"""
        if not isinstance(x_dct, torch.Tensor):
            raise TypeError("Input must be torch.Tensor")

        B, C, H, W = x_dct.shape
        pad_h = (self.block_size - H % self.block_size) % self.block_size
        pad_w = (self.block_size - W % self.block_size) % self.block_size
        if pad_h or pad_w:
            x_dct = F.pad(x_dct, (0, pad_w, 0, pad_h), mode='reflect')
        padded_h, padded_w = x_dct.shape[-2:]

        weights = self.weights.to(x_dct.device)
        weights = weights.view(1, 1, self.block_size, self.block_size)
        weights = weights.repeat(B, C, padded_h // self.block_size, padded_w // self.block_size)
        return (x_dct * weights)[:, :, :H, :W]


class NVFLayer(nn.Module):
    """噪声可见度函数（基于局部方差）"""

    def __init__(self, window_size=7, D=75):
        super().__init__()
        self.pool = nn.AvgPool2d(window_size, stride=1, padding=window_size // 2)
        self.D = D

    def forward(self, x):
        local_mean = self.pool(x)
        local_var = self.pool(x**2) - local_mean**2

        sigma_x_max_sq = torch.max(local_var)
        theta = self.D / (sigma_x_max_sq + 1e-6)
        return 1 / (1 + theta * local_var)


class DCTExtractor_withCSF(nn.Module):

    def __init__(self, block_size=8, swin_dims=[256, 512, 1024, 2048], ppd=30.0, nvf_window_size=7, nvf_D=75):
        super().__init__()
        target_strides = [1, 2, 4, 4]
        self.block_size = block_size
        self.csf = CSFWeightGenerator_BT(block_size, ppd=ppd)
        self.nvf = NVFLayer(window_size=nvf_window_size, D=nvf_D)

        self.proj_layers = nn.ModuleList(
            [self.build_projection(dim, stride) for dim, stride in zip(swin_dims, target_strides)])
        self.downsample_filter = DownSampleFilter()
        self.dense_block = DenseBlock()

        self.conv_3to64Channel = nn.Conv2d(3, 64, kernel_size=1, stride=1)
        self.conv_downChannel384to64 = nn.Conv2d(384, 64, kernel_size=1, stride=1)

    def build_projection(self, dim, stride):
        """
            构建投影层
            dim: 目标通道数
            stride: 空间下采样步长 (1表示大小不变, 2表示宽高减半)
            """
        return nn.Sequential(
            nn.Conv2d(64, dim // 2, kernel_size=3, stride=stride, padding=1), nn.GELU(),
            nn.Conv2d(dim // 2, dim, kernel_size=3, padding=1))

    def dct2d(self, x):
        """对图像块执行2D DCT（分块处理）"""
        B, C, H, W = x.shape
        pad_h = (self.block_size - H % self.block_size) % self.block_size
        pad_w = (self.block_size - W % self.block_size) % self.block_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        padded_h, padded_w = x.shape[-2:]

        x_blocks = x.unfold(2, self.block_size, self.block_size).unfold(3, self.block_size, self.block_size)
        x_blocks = x_blocks.contiguous().view(B, C, -1, self.block_size, self.block_size)

        dct_blocks = []
        for i in range(x_blocks.shape[2]):
            block = x_blocks[:, :, i]
            block_dct = torch.tensor(
                dct(dct(block.detach().cpu().numpy(), axis=3, norm='ortho'), axis=2, norm='ortho'), device=x.device)
            dct_blocks.append(block_dct)

        x_dct = torch.stack(dct_blocks, dim=2).view(B, C, padded_h, padded_w)
        return x_dct[:, :, :H, :W]

    def forward(self, x):

        nvf_map = self.nvf(x.mean(dim=1, keepdim=True))
        x = x * nvf_map

        x = self.conv_3to64Channel(x)
        B, C, H, W = x.shape

        out = []
        for i, proj in enumerate(self.proj_layers):
            x = self.downsample_filter(x)

            x_384 = self.dense_block(x)
            x_64 = self.conv_downChannel384to64(x_384)

            x_dct = self.dct2d(x_64)

            x_dct_weighted = self.csf(x_dct)

            x_feat = self.downsample_filter(self.downsample_filter(x_dct_weighted))
            proj_feat = proj(x_feat)

            out.append(proj_feat)

        return out


@ARCH_REGISTRY.register()
class CFANet(nn.Module):

    def __init__(self,
                 semantic_model_name='swin_base_patch4_window7_224_in22k',
                 model_name='uiqa_ava_swin_t',
                 backbone_pretrain=True,
                 in_size=None,
                 use_ref=False,
                 num_class=1,
                 num_crop=1,
                 crop_size=256,
                 inter_dim=256,
                 num_heads=4,
                 num_attn_layers=1,
                 dprate=0.1,
                 activation='gelu',
                 pretrained=False,
                 pretrained_model_path=None,
                 out_act=False,
                 block_pool='weighted_avg',
                 test_img_size=None,
                 align_crop_face=True,
                 default_mean=IMAGENET_DEFAULT_MEAN,
                 default_std=IMAGENET_DEFAULT_STD,
                 use_dwt=False,
                 use_lbf=False,
                 psam_window_size=7,
                 nvf_D=75,
                 pfem_dct_block_size=8,
                 pfem_ppd=30.0):
        super().__init__()

        self.in_size = in_size

        self.model_name = model_name
        self.semantic_model_name = semantic_model_name
        self.semantic_level = -1
        self.crop_size = crop_size
        self.use_ref = use_ref

        self.num_class = num_class
        self.block_pool = block_pool
        self.test_img_size = test_img_size

        self.align_crop_face = align_crop_face

        self.use_dwt = use_dwt
        self.dwt = DCTExtractor_withCSF(
            block_size=pfem_dct_block_size, ppd=pfem_ppd, nvf_window_size=psam_window_size, nvf_D=nvf_D)

        self.nvf = NVFLayer(window_size=psam_window_size, D=nvf_D)

        if 'swin' not in semantic_model_name.lower():
            raise ValueError(f'PINE requires a Swin backbone, got {semantic_model_name!r}.')
        self.semantic_model = create_swin(semantic_model_name, pretrained=backbone_pretrain, drop_path_rate=0.0)
        feature_dim = self.semantic_model.num_features
        feature_dim_list = [int(self.semantic_model.embed_dim * 2**i) for i in range(self.semantic_model.num_layers)]
        feature_dim_list = feature_dim_list[1:] + [feature_dim]

        self.default_mean = torch.Tensor(default_mean).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(default_std).view(1, 3, 1, 1)

        self.fusion_mul = 3 if use_ref else 1
        ca_layers = sa_layers = num_attn_layers

        self.act_layer = nn.GELU() if activation == 'gelu' else nn.ReLU()
        dim_feedforward = min(4 * inter_dim, 2048)

        self.conv2 = nn.ModuleList()

        proj_dims = [256, 512, 1024, 2048]
        for idx, dim in enumerate(feature_dim_list):
            dim = dim * 3 if use_ref else dim
            in_channels = dim + proj_dims[idx]

            self.conv2.append(
                nn.Sequential(nn.Conv2d(in_channels, dim, kernel_size=1, stride=1, padding=0), self.act_layer))

        tmp_layer = TransformerEncoderLayer(
            inter_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            normalize_before=True,
            dropout=dprate,
            activation=activation)

        self.dim_reduce = nn.ModuleList()
        self.weight_pool = nn.ModuleList()
        for idx, dim in enumerate(feature_dim_list):
            dim = dim * 3 if use_ref else dim
            if use_ref:
                self.weight_pool.append(
                    nn.Sequential(
                        nn.Conv2d(dim // 3, 64, 1, stride=1), self.act_layer, nn.Conv2d(64, 64, 3, stride=1, padding=1),
                        self.act_layer, nn.Conv2d(64, 1, 3, stride=1, padding=1), nn.Sigmoid()))
            else:
                self.weight_pool.append(GatedConv(dim))

            self.dim_reduce.append(nn.Sequential(
                nn.Conv2d(dim, inter_dim, 1, 1),
                self.act_layer,
            ))

        self.attn_blks = nn.ModuleList()

        tmp_layer = TransformerDecoderLayer(
            inter_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            normalize_before=True,
            dropout=dprate,
            activation=activation)
        for i in range(len(feature_dim_list) - 1):
            self.attn_blks.append(TransformerDecoder(tmp_layer, ca_layers))

        self.attn_pool = TransformerEncoderLayer(
            inter_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            normalize_before=True,
            dropout=dprate,
            activation=activation)

        linear_dim = inter_dim

        if out_act and self.num_class == 1:
            self.score_linear.append(nn.Softplus())

        if self.num_class > 1:
            self.score_linear.append(nn.Softmax(dim=-1))

        self.h_emb = nn.Parameter(torch.randn(1, inter_dim // 2, 32, 1))
        self.w_emb = nn.Parameter(torch.randn(1, inter_dim // 2, 1, 32))

        nn.init.trunc_normal_(self.h_emb.data, std=0.02)
        nn.init.trunc_normal_(self.w_emb.data, std=0.02)
        self._init_linear(self.dim_reduce)

        self._init_linear(self.attn_blks)
        self._init_linear(self.attn_pool)

        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, False, weight_keys='params')

        self.eps = 1e-8
        self.crops = num_crop

    def _init_linear(self, m):
        for module in m.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                nn.init.constant_(module.bias.data, 0)

    def preprocess(self, x):
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        return x

    def fix_bn(self, model):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad = False
                m.eval()

    def get_swin_feature(self, model, x):
        b, c, h, w = x.shape
        x = model.patch_embed(x)
        if model.absolute_pos_embed is not None:
            x = x + model.absolute_pos_embed
        x = model.pos_drop(x)
        feat_list = []
        for ly in model.layers:
            x = ly(x)
            feat_list.append(x)

        h, w = h // 8, w // 8
        for idx, f in enumerate(feat_list):
            feat_list[idx] = f.transpose(1, 2).reshape(b, f.shape[-1], h, w)
            if idx < len(feat_list) - 2:
                h, w = h // 2, w // 2

        return feat_list

    def dist_func(self, x, y, eps=1e-12):
        return torch.sqrt((x - y)**2 + eps)

    def freeze_swin_layers(model, num_frozen_stages=2):
        model.patch_embed.requires_grad_(False)

        for i in range(num_frozen_stages):
            layer = model.layers[i]
            for param in layer.parameters():
                param.requires_grad = False

    def forward_cross_attention(self, x, y=None):

        if not self.training:
            _, _, height, width = x.shape
            target_size = 384
            pad_h = max(target_size - height, 0)
            pad_w = max(target_size - width, 0)
            pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            x = F.pad(x, pad, mode='reflect')

        x = self.preprocess(x)
        if self.use_ref:
            y = self.preprocess(y)

        if self.use_dwt == True:

            dwt_feats = self.dwt(x)

        dist_feat_list = self.get_swin_feature(self.semantic_model, x)
        if self.use_ref:
            ref_feat_list = self.get_swin_feature(self.semantic_model, y)
        self.semantic_model.eval()

        start_level = 0
        end_level = len(dist_feat_list)

        b, c, th, tw = dist_feat_list[end_level - 1].shape
        pos_emb = torch.cat(
            (self.h_emb.repeat(1, 1, 1, self.w_emb.shape[3]), self.w_emb.repeat(1, 1, self.h_emb.shape[2], 1)), dim=1)

        token_feat_list = []

        for i in reversed(range(start_level, end_level)):
            tmp_dist_feat = dist_feat_list[i]

            if self.use_dwt == True and i <= 2:
                dwt_dist_feat = dwt_feats[i]

                target_size = tmp_dist_feat.shape[2:]
                if dwt_dist_feat.shape[2:] != target_size:
                    dwt_dist_feat = F.interpolate(dwt_dist_feat, size=target_size, mode='bilinear', align_corners=False)

                tmp_dist_feat = torch.cat([tmp_dist_feat, dwt_dist_feat], dim=1)
                tmp_dist_feat = self.conv2[i](tmp_dist_feat)

            if self.use_ref:
                tmp_ref_feat = ref_feat_list[i]
                diff = self.dist_func(tmp_dist_feat, tmp_ref_feat)

                tmp_feat = torch.cat([tmp_dist_feat, tmp_ref_feat, diff], dim=1)
                weight = self.weight_pool[i](diff)
                tmp_feat = tmp_feat * weight
            else:
                tmp_feat = self.weight_pool[i](tmp_dist_feat)

            if tmp_feat.shape[2] > th and tmp_feat.shape[3] > tw:
                tmp_feat = F.adaptive_avg_pool2d(tmp_feat, (th, tw))

            tmp_feat = self.dim_reduce[i](tmp_feat)
            tmp_feat = tmp_feat.flatten(2).permute(2, 0, 1)

            token_feat_list.append(tmp_feat)

        query = token_feat_list[0]
        query_list = [query]
        for i in range(len(token_feat_list) - 1):
            key_value = token_feat_list[i + 1]

            query = self.attn_blks[i](query, key_value)

            query_list.append(query)

        return query

    def forward(self, x, y=None, return_mos=True, return_dist=False):
        if self.use_ref:
            assert y is not None, f'Please input y when use reference is True.'
        else:
            y = None

        if self.crops > 1 and not self.training:
            bsz = x.shape[0]
            if y is not None:
                x, y = uniform_crop([x, y], self.crop_size, self.crops)
            else:
                x = uniform_crop([x], self.crop_size, self.crops)[0]

            score = self.forward_cross_attention(x, y)
            score = score.reshape(bsz, self.crops, self.num_class)
            score = score.mean(dim=1)
        else:
            score = self.forward_cross_attention(x, y)

        return_list = []

        return_list.append(score)

        if len(return_list) > 1:
            return return_list
        else:
            return return_list[0]


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Model_SwinT(nn.Module):

    def __init__(self, semantic_model_name, pretrained=False, pretrained_model_path=None, use_dwt=False, **kwargs):
        super(Model_SwinT, self).__init__()

        model = CFANet(
            semantic_model_name,
            pretrained=pretrained,
            pretrained_model_path=pretrained_model_path,
            use_dwt=use_dwt,
            **kwargs)
        model.head = Identity()

        self.feature_extraction = model

    def quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),
        )

        return regression_block

    def forward(self, x):

        x = self.feature_extraction(x)

        return x


@ARCH_REGISTRY.register()
class UFANet(nn.Module):

    def __init__(self,
                 semantic_model_name="swin_base_patch4_window12_384_in22k",
                 pretrained=False,
                 drop=0.1,
                 pretrained_model_path=None,
                 **kwargs):

        super().__init__()

        swin_t_distortion = Model_SwinT(
            semantic_model_name,
            pretrained=pretrained,
            pretrained_model_path=pretrained_model_path,
            use_dwt=True,
            **kwargs)

        self.distortion_feature_extraction = swin_t_distortion.feature_extraction

        self.attn_pool_d = TransformerEncoderLayer(256, nhead=4)

        self.quality_d = self.quality_regression(256, 1, drop)

    def quality_regression(self, in_channels, out_channels, drop):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 16), nn.GELU(), nn.Linear(16, out_channels), nn.Sigmoid())

        return regression_block

    def forward(self, x_distortion):

        x_distortion = self.distortion_feature_extraction(x_distortion)

        x_distortion = self.attn_pool_d(x_distortion)
        mos_distortion = self.quality_d(x_distortion.mean(dim=0))

        return mos_distortion, mos_distortion
