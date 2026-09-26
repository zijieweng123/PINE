import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from torch.nn import functional as F

from pyiqa.archs import build_network
from pyiqa.losses import build_loss
from pyiqa.metrics import calculate_metric
from pyiqa.utils import get_root_logger, imwrite, tensor2img
from pyiqa.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import pickle
import numpy as np

@MODEL_REGISTRY.register()
class HRIQAModel(BaseModel):
    """General module to train an IQA network."""

    def __init__(self, opt):
        super(HRIQAModel, self).__init__(opt)

        # define network
        self.net = build_network(opt['network'])
        self.net = self.model_to_device(self.net)
        self.print_network(self.net)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net, load_path, self.opt['path'].get('strict_load', True), param_key)

        if self.is_train:
            self.init_training_settings()



    def init_training_settings(self):
        self.net.train()
        train_opt = self.opt['train']

        self.net_best = build_network(self.opt['network']).to(self.device)


        # define losses
        if train_opt.get('mos_loss_opt'):
            self.cri_mos = build_loss(train_opt['mos_loss_opt']).to(self.device)
        else:
            self.cri_mos = None

        # define metric related loss, such as plcc loss
        if train_opt.get('metric_loss_opt'):
            self.cri_metric = build_loss(train_opt['metric_loss_opt']).to(self.device)
        else:
            self.cri_metric = None

        if train_opt.get('class_loss_opt'):
            self.cri_class = build_loss(train_opt['class_loss_opt']).to(self.device)
        else:
            self.cri_class = None

        # define metric related 
        
        if train_opt.get('nce_loss_opt'):
            self.cri_nce = build_loss(train_opt['nce_loss_opt']).to(self.device)
            pkl_path = self.opt['train']['global_pkl_path']  
            data = pickle.load(open(pkl_path, 'rb'))

            feats_np = data['feats']
            self.global_mos = data['mos']               # numpy (N,)
            self.global_names = data['names']           # numpy (N,), dtype='<U...'
            
            self.global_feats = torch.from_numpy(feats_np).float().to(self.device)
            self.worst5_idxs = np.argsort(self.global_mos)[:5]
            
        else:
            self.cri_nce = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_opt = train_opt['optim']

        # for param in self.net.parameters():
        #     param.requires_grad = True

        # blacklist_layers = [
        #     'semantic_model' 
        # ]

        # # 3. 遍历网络参数，只要名字里包含了黑名单的关键词，就切断梯度
        # for name, param in self.net.named_parameters():
        #     if any(layer_name in name for layer_name in blacklist_layers):
        #         param.requires_grad = False

        # # 4. 打印参数统计，确认庞大的主干确实被冻结了
        # trainable_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        # frozen_params = sum(p.numel() for p in self.net.parameters() if not p.requires_grad)

        # print(f"Frozen Parameters (e.g., Backbone): {frozen_params:,}")
        # print(f"Trainable Parameters (e.g., RAG & Heads): {trainable_params:,}")

        # for param in self.net.parameters():
        #     param.requires_grad = False

        # trainable_layers = [

        #     'rag_projs',         
        #     'label_embeds',      
        #     'rag_attn_blks',    
        #     'attn_blks',         
        #     # 'ccff_atten',    
        #     # 'cga_atten', 
        #     'score_mlps',
        #     'fusion_projs',
        #     'attn_pool',
        #     'score_linear',


        #     'attn_pool_d',     
        #     'quality_d'         
        # ]

        # for name, param in self.net.named_parameters():
        #     if any(layer_name in name for layer_name in trainable_layers):
        #         param.requires_grad = True
                
        param_dict = {k: v for k, v in self.net.named_parameters()}
        param_keys = list(param_dict.keys())
        # set different lr for different modules if needed, e.g., lr_backbone, lr_head
        lr_keys = [i for i in optim_opt.keys() if i.startswith('lr_')]

        optim_params = []
        for key in lr_keys:
            if key.startswith('lr_'):
                module_key = key.replace('lr_', '')
                logger = get_root_logger()
                logger.info(f'Set optimizer for {module_key} with lr: {optim_opt[key]}, weight_decay: {optim_opt.get(f"weight_decay_{module_key}", 0.)}')

                optim_params.append({
                    'params': [param_dict[k] for k in param_keys if module_key in k and param_dict[k].requires_grad],
                    'lr': optim_opt.pop(key, 0.),
                    'weight_decay': optim_opt.pop(f'weight_decay_{module_key}', 0.),
                })

                # should use param_keys[:] to avoid iteration error
                for k in param_keys[:]:
                    if module_key in k:
                        param_keys.remove(k)
        
        # append the rest of the parameters
        optim_params.append({
            'params': [param_dict[k] for k in param_keys if param_dict[k].requires_grad],
        })

        # log params that will not be optimized
        for k, v in param_dict.items():
            if not v.requires_grad:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        
        # remove blank param list
        for k in optim_params:
            if len(k['params']) == 0:
                optim_params.remove(k)

        optim_type = train_opt['optim'].pop('type')
        self.optimizer = self.get_optimizer(optim_type, optim_params, **train_opt['optim'])
        self.optimizers.append(self.optimizer)
    
    def feed_data(self, data):
        # self.img_aesthetics = data['img_aesthetics'].to(self.device)
        self.img_distortion = data['img_distortion'].to(self.device)
        # self.img_saliency = data['img_saliency'].to(self.device)
        # self.img_dtcwt = data['img_dtcwt'].to(self.device)

        if 'mos_label' in data:
            self.gt_mos = data['mos_label'].to(self.device)

        if 'cls_label' in data:
            self.gt_cls = data['cls_label'].to(self.device)
            
        if 'ref_img' in data:
            self.use_ref = True
            self.ref_input = data['ref_img'].to(self.device)
        else:
            self.use_ref = False

        if 'use_ref' in self.opt['train']:
            self.use_ref = self.opt['train']['use_ref']
        
        # ==========================================
        # [新增] 接收 RAG 数据分支
        # ==========================================
        if 'retrieved_feats' in data:
            # self.patches = data['patches'].to(self.device)
            self.retrieved_labels = data['retrieved_labels'].to(self.device)
            self.retrieved_feats = data['retrieved_feats'].to(self.device)
            self.use_rag_branch = True
        else:
            self.patches = None
            self.retrieved_labels = None
            self.retrieved_feats = None
            self.use_rag_branch = False
        
        if 'rag_data' in data:
            self.rag_data = data['rag_data']
            self.use_rag_data = True
        else:
            self.use_rag_data = False
            
    def _net_forward_single(self, net, img_distortion, ref_input=None):
        if self.use_rag_branch:
            return net(img_distortion, retrieved_feats=self.retrieved_feats, retrieved_labels=self.retrieved_labels)
        
        if self.use_rag_data:
            return net(img_distortion, rag_data=self.rag_data)
            
        if self.use_ref:
            return net(img_distortion, ref_input)
        else:
            return net(img_distortion)

    def net_forward(self, net):
        # Multi-patch validation inputs have shape [B, N, C, H, W]. Process
        # only a small number of patches per forward pass and average scores
        # over N, avoiding a B*N GPU memory spike.
        if self.img_distortion.dim() != 5:
            return self._net_forward_single(net, self.img_distortion, getattr(self, 'ref_input', None))

        if self.use_rag_branch or self.use_rag_data or self.use_ref:
            raise ValueError('Multi-patch preprocessing currently supports no-reference HNR inputs only.')

        batch_size, patch_count = self.img_distortion.shape[:2]
        crop_batch_size = int(self.opt.get('multi_crop_batch_size', 1))
        if crop_batch_size < 1:
            raise ValueError('network.multi_crop_batch_size must be >= 1.')

        score_chunks = []
        cls_chunks = []
        for start in range(0, patch_count, crop_batch_size):
            end = min(start + crop_batch_size, patch_count)
            patch_batch = self.img_distortion[:, start:end].reshape(
                batch_size * (end - start), *self.img_distortion.shape[2:])
            output_score, output_cls = self._net_forward_single(net, patch_batch)
            score_chunks.append(output_score.reshape(batch_size, end - start, -1))
            cls_chunks.append(output_cls.reshape(batch_size, end - start, -1))

        return torch.cat(score_chunks, dim=1).mean(dim=1), torch.cat(cls_chunks, dim=1).mean(dim=1)

    def optimize_parameters(self, current_iter, accumulate=False):
        # if not accumulate:
        #     self.optimizer.zero_grad(set_to_none=True)

        # self.output_score = self.net_forward(self.net)
        self.output_score, self.output_cls = self.net_forward(self.net)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_mos:
            l_mos = self.cri_mos(self.output_score, self.gt_mos)
            l_total += l_mos
            loss_dict['l_mos'] = l_mos

        if self.cri_metric:
            l_metric = self.cri_metric(self.output_score, self.gt_mos)
            # l_metric = self.cri_metric(self.output_cls, self.gt_cls)
            l_total += l_metric
            loss_dict['l_metric'] = l_metric

        if self.cri_class:
            # l_metric = self.cri_metric(self.output_score, self.gt_mos)
            l_class = self.cri_class(self.output_cls, self.gt_cls)
            l_total += l_class
            loss_dict['l_class'] = l_metric
            
        if self.cri_nce:
            info_nce_total = 0
            B = self.x_feat.size(0)
            # normalize global features 
            norm_global_feats = F.normalize(self.global_feats, dim=1).squeeze(1)  # [N, D]

            for i in range(B):
                anchor_feat = self.x_feat[i].unsqueeze(0)  # [1, D]
                anchor_feat_norm = F.normalize(anchor_feat, dim=1)     # [1, D]
                anchor_mos = self.gt_mos[i].item()
        

                # Compute cosine distances with all global features
                cos_sims = torch.matmul(anchor_feat_norm, norm_global_feats.T).squeeze(0)  # [N]
                cos_dists = 1 - cos_sims  # 越小越相似

                # ------ 正样本选择 ------
                mos_mask = self.global_mos > anchor_mos  # boolean mask
                valid_pos_indices = torch.where(torch.from_numpy(mos_mask))[0]
                if len(valid_pos_indices) == 0:
        
                    pos_idx = torch.argmax(self.global_mos).item()
                else:
                    valid_pos_dists = cos_dists[valid_pos_indices]
                    pos_idx = valid_pos_indices[torch.argmin(valid_pos_dists)].item()

                positive = self.global_feats[pos_idx].unsqueeze(0).to(self.device)  # [1, D]

                # ------ 负样本选择 ------
                mos_mask_neg = self.global_mos < anchor_mos
                valid_neg_indices = torch.where(torch.from_numpy(mos_mask_neg))[0]
                if len(valid_neg_indices) < 5:
            
                    neg_indices = torch.randperm(len(self.global_feats))[:5]
                else:
                    valid_neg_dists = cos_dists[valid_neg_indices]
                    
                    sorted_idx = torch.argsort(valid_neg_dists)[:5].to(self.device)
                    neg_indices = valid_neg_indices[sorted_idx]

                neg_feats = self.global_feats[neg_indices].to(self.device)  # [5, D]

                # ------ NCE Loss ------
                loss_i = self.cri_nce(anchor_feat, positive, neg_feats)
                nce_total += loss_i

            # for i in range(B):
            #     anchor = self.x_feat[i]  
            #     anchor = anchor.unsqueeze(0)
            #     positive = self.positive_feat.to(anchor.device)  # 固定正样本
            #     negatives = self.x_feat[torch.arange(B) != i]  # batch 内其他图作为负样本

            #     loss_i = self.cri_nce(anchor, positive, negatives)
            #     info_nce_total += loss_i

            # for i in range(B):
            #     anchor_feat = self.x_feat[i].unsqueeze(0)  # [1, D]
            #     anchor_mos = self.gt_mos[i].item()

            #     larger = np.where(self.global_mos > anchor_mos)[0]
            #     if len(larger)==0:
            #         pos_idx = int(np.argmax(self.global_mos))
            #     else:
            #         diffs = self.global_mos[larger] - anchor_mos
            #         pos_idx = int(larger[np.argmin(diffs)])

            #     positive = self.global_feats[pos_idx].to(self.device)  # [1, D]

            #     neg_feats = self.global_feats[self.worst5_idxs].to(self.device).squeeze(1)      # [5, D]

            #     loss_i = self.cri_nce(anchor_feat, positive, neg_feats)
            #     info_nce_total += loss_i

            l_nce = info_nce_total / B
            l_total += l_nce
            loss_dict['l_nce'] = l_nce


        # normalize loss for accumulation
        accumulation_steps = self.opt['train'].get('accumulation_steps')
        l_total = l_total / accumulation_steps

        l_total.backward()
        if not accumulate:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        self.log_dict = self.reduce_loss_dict(loss_dict)

        # log metrics in training batch
        pred_score = self.output_score.squeeze(1).cpu().detach().numpy()
        gt_mos = self.gt_mos.squeeze(1).cpu().detach().numpy()
        for name, opt_ in self.opt['val']['metrics'].items():
            self.log_dict[f'train_metrics/{name}'] = calculate_metric([pred_score, gt_mos], opt_)

    # def optimizer_step(self):
    #     self.optimizer.step()
    #     self.optimizer.zero_grad()

    # def optimize_parameters(self, current_iter):
    #     self.optimizer.zero_grad()
    #     self.output_score = self.net_forward(self.net)
    #     # self.output_score, score2, score3 = self.net_forward(self.net)

    #     l_total = 0
    #     loss_dict = OrderedDict()
    #     # pixel loss
    #     if self.cri_mos:
    #         l_mos = self.cri_mos(self.output_score, self.gt_mos)
    #         l_total += l_mos

    #         # l_mos_2 = self.cri_mos(score2, self.gt_mos)
    #         # l_total += l_mos_2 * 0.25
    #         # l_mos_3 = self.cri_mos(score3, self.gt_mos)
    #         # l_total += l_mos_3 * 0.25
            
    #         loss_dict['l_mos'] = l_mos


    #     if self.cri_metric:
    #         l_metric = self.cri_metric(self.output_score, self.gt_mos)
    #         l_total += l_metric
    #         loss_dict['l_metric'] = l_metric

    #     l_total.backward()
    #     self.optimizer.step()

    #     self.log_dict = self.reduce_loss_dict(loss_dict)

    #     # log metrics in training batch
    #     pred_score = self.output_score.squeeze(1).cpu().detach().numpy()
    #     gt_mos = self.gt_mos.squeeze(1).cpu().detach().numpy()
    #     for name, opt_ in self.opt['val']['metrics'].items():
    #         self.log_dict[f'train_metrics/{name}'] = calculate_metric([pred_score, gt_mos], opt_)

    def test(self):
        self.net.eval()
        with torch.no_grad():
            # if self.img_saliency.dim() == 5:  # [B, 5, 3, H, W]
            if False:
                B, N, C, H, W = self.img_saliency.shape  # N=5
                max_batch_size = self.img_saliency.size(0)
                
                # 展开所有patch为大batch
                saliency_all = self.img_saliency.view(B * N, C, H, W).to(self.device)
                aesthetics_all = self.img_aesthetics.unsqueeze(1).expand(-1, N, -1, -1, -1).reshape(B * N, C, H, W).to(self.device)
                distortion_all = self.img_distortion.unsqueeze(1).expand(-1, N, -1, -1, -1).reshape(B * N, C, H, W).to(self.device)
                
                outputs = []
                total = B * N
                for start in range(0, total, max_batch_size):
                    end = min(start + max_batch_size, total)
                    out_batch, out_cls = self.net(
                        aesthetics_all[start:end],
                        distortion_all[start:end],
                        saliency_all[start:end]
                    )
                    outputs.append(out_batch)
                
                output_all = torch.cat(outputs, dim=0)  # [B*N, 1]
                
                # reshape为 [B, N, 1]
                output_all = output_all.view(B, N, -1)
                # output_all: [B, N=5, 1]
                weights = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.6], device=output_all.device).view(1, 5, 1)  # 权重和为1
                self.output_score = (output_all * weights).sum(dim=1)  # [B, 1]

                # 每个样本5个patch取平均
                # self.output_score = output_all.mean(dim=1)  # [B, 1]
    
            else:
                self.output_score, self.output_cls = self.net_forward(self.net)
        self.net.train()



    def net_forward_single(self, single_saliency):
        if self.use_ref:
            # return self.net(self.img_aesthetics, self.img_distortion, single_saliency, self.ref_input)
            return self.net(self.img_distortion, self.ref_input)
        else:
            # return self.net(self.img_aesthetics, self.img_distortion, single_saliency)
            return self.net(self.img_distortion)
    
        
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        pred_score = []
        gt_mos = []
        for idx, val_data in enumerate(dataloader):
            img_name = osp.basename(val_data['img_path'][0])
            self.feed_data(val_data)
            self.test()
            pred_score.append(self.output_score)
            gt_mos.append(self.gt_mos)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name:>20}')
        if use_pbar:
            pbar.close()

        pred_score = torch.cat(pred_score, dim=0).squeeze(1).cpu().numpy()
        gt_mos = torch.cat(gt_mos, dim=0).squeeze(1).cpu().numpy()
        repeat_times = dataloader.dataset.opt.get('repeat_times', 1)
        if repeat_times > 1:
            new_len = len(pred_score) // repeat_times
            pred_score = pred_score.reshape(new_len, repeat_times).mean(axis=1)
            gt_mos = gt_mos[::repeat_times]
            
        if with_metrics:
            # calculate all metrics
            for name, opt_ in self.opt['val']['metrics'].items():
                self.metric_results[name] = calculate_metric([pred_score, gt_mos], opt_)

            if self.key_metric is not None:
                # If the best metric is updated, update and save best model
                to_update = self._update_best_metric_result(dataset_name, self.key_metric,
                                                            self.metric_results[self.key_metric], current_iter)

                if to_update:
                    for name, opt_ in self.opt['val']['metrics'].items():
                        self._update_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                    self.copy_model(self.net, self.net_best)
                    self.save_network(self.net_best, 'net_best')
            else:
                # update each metric separately
                updated = []
                for name, opt_ in self.opt['val']['metrics'].items():
                    tmp_updated = self._update_best_metric_result(dataset_name, name, self.metric_results[name],
                                                                  current_iter)
                    updated.append(tmp_updated)
                # save best model if any metric is updated
                if sum(updated):
                    self.copy_model(self.net, self.net_best)
                    self.save_network(self.net_best, 'net_best')

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'val_metrics/{dataset_name}/{metric}', value, current_iter)

    def save(self, epoch, current_iter, save_net_label='net'):
        self.save_network(self.net, save_net_label, current_iter)
        self.save_training_state(epoch, current_iter)
