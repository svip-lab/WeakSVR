import argparse
import os
import random
import time
import warnings
import yaml
import numpy as np
import torch
import torch.distributed as dist
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from sklearn.metrics import auc, roc_curve, accuracy_score,top_k_accuracy_score,recall_score
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm
import torch.nn as nn
from Data.clip_dataset_retrival import load_dataset_retrival_ddp
from configs.defaults import get_cfg_defaults
from models.build_model import build_model,update_logit_scale
from utils.logger import setup_logger
from utils.loss import compute_cls_loss, compute_seq_loss, compute_info_loss_neg, compute_gumbel_loss
from collections import OrderedDict
from utils.metrics import compute_WDR, pred_dist
from utils.preprocess import frames_preprocess
from utils.utils_distributed import all_gather_concat,all_reduce_mean,all_reduce_sum,all_gather_object
warnings.filterwarnings("ignore")


# torch.autograd.set_detect_anomaly(True)
def setup(local_rank):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')


def init_log(cfg=None, eval_cfg=None, args=None, local_rank=0,):

    # cfg, eval_cfg = update_cfg_from_args(cfg, eval_cfg, args)
    logger_path = os.path.join(eval_cfg.TRAIN.SAVE_PATH, args.tensorboard + '/logs')
    logger = setup_logger('Sequence Verification', logger_path, args.log_name, args.local_rank)
    if args.eval:
        logger.info('-------------Update eval cfg from train config-------------\n')
        logger.info('Running eval with config:\n{}\n'.format(eval_cfg))



    return logger


def eval_retrival(cfg, eval_cfg, args,):

    local_rank = args.local_rank
    setup(local_rank)
    setup_seed(cfg.TRAIN.SEED + local_rank)
    logger = init_log(cfg, eval_cfg, args, local_rank)

    model = build_model(cfg=cfg, args=args, model_log=False,).to(local_rank)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Load checkpoint

    if args.load_path and os.path.isfile(args.load_path):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(args.load_path, map_location=map_location)
        # new_state_dict = OrderedDict()
        # for k, v in checkpoint['model_state_dict'].items():
        #     name = 'module.' + k
        #     new_state_dict[name] = v
        # model.load_state_dict(new_state_dict, strict=True)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)

        logger.info('-> Loaded checkpoint %s' % (args.load_path))
    else:
        raise IOError('no ckpt has been load')

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    test_loader = load_dataset_retrival_ddp(eval_cfg, args, drop_last=False)

    start_time = time.time()
    # eval
    model.eval()
    # -------------------------------------------
    # test on val set
    with torch.no_grad():
        labels, preds, pred2_top2, labels1_all, labels2_all = None, None, None, None, None
        if dist.get_rank() == 0:
            iter_test = tqdm(test_loader)
        else:
            iter_test = test_loader
        for iter, sample in enumerate(iter_test):
            if iter == 3 and args.debug:
                break
            frames1_list = sample['clips1']
            frames2_list = sample['clips2']
            frames3_list = sample['clips3']
            frames4_list = sample['clips4']
            frames5_list = sample['clips5']

            label = sample['label'].to(local_rank, non_blocking=True)
            label_token1 = sample['label_token1'].to(local_rank, non_blocking=True)

            embeds1_list = []
            embeds2_list = []
            embeds3_list = []
            embeds4_list = []
            embeds5_list = []

            for i in range(len(frames1_list)):
                frames1 = frames_preprocess(frames1_list[i]).to(local_rank, non_blocking=True)
                frames2 = frames_preprocess(frames2_list[i]).to(local_rank, non_blocking=True)
                frames3 = frames_preprocess(frames3_list[i]).to(local_rank, non_blocking=True)
                frames4 = frames_preprocess(frames4_list[i]).to(local_rank, non_blocking=True)
                frames5 = frames_preprocess(frames5_list[i]).to(local_rank, non_blocking=True)
                embeds1, text_features = model(frames1, text_token=label_token1, embed=True, retrieval=True)
                embeds2 = model(frames2, embed=True)
                embeds3 = model(frames3, embed=True)
                embeds4 = model(frames4, embed=True)
                embeds5 = model(frames5, embed=True)

                embeds1_list.append(embeds1.unsqueeze(dim=0))
                embeds2_list.append(embeds2.unsqueeze(dim=0))
                embeds3_list.append(embeds3.unsqueeze(dim=0))
                embeds4_list.append(embeds4.unsqueeze(dim=0))
                embeds5_list.append(embeds5.unsqueeze(dim=0))

            embeds1_avg = (torch.cat(embeds1_list, dim=0)).mean(dim=0).unsqueeze(dim=1)
            embeds2_avg = (torch.cat(embeds2_list, dim=0)).mean(dim=0).unsqueeze(dim=1)
            embeds3_avg = (torch.cat(embeds3_list, dim=0)).mean(dim=0).unsqueeze(dim=1)
            embeds4_avg = (torch.cat(embeds4_list, dim=0)).mean(dim=0).unsqueeze(dim=1)
            embeds5_avg = (torch.cat(embeds5_list, dim=0)).mean(dim=0).unsqueeze(dim=1)
            image_features = torch.cat([embeds1_avg, embeds2_avg, embeds3_avg, embeds4_avg,embeds5_avg], dim=1)
            text_features = text_features.unsqueeze(dim=1)
            image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
            pre_text = text_features @ image_features.permute(0, 2, 1).squeeze(dim=-1)
            pred_top2 = pre_text.squeeze()
            pred = torch.argmax(pre_text, dim=-1).squeeze()
            # pred = pred_dist(args.dist, embeds1_avg, embeds2_avg)

            torch.cuda.synchronize()

            # gather from other gpu

            pred = all_gather_concat(pred)
            pred_top2 = all_gather_concat(pred_top2)
            label = all_gather_concat(label)

            # add all data to list
            if iter == 0:
                preds = pred
                preds_top2 = pred_top2
                labels = label

            else:
                preds = torch.cat([preds, pred])
                preds_top2 = torch.cat([preds_top2, pred_top2])
                labels = torch.cat([labels, label])
    # if dist.get_rank()==0:
    #     print('preds:',preds)
    #     print('preds:',preds.shape)
    #     print('labels:',labels)
    #     print('labels:',labels.shape)
    labels = labels.cpu().detach().numpy()
    preds = preds.cpu().detach().numpy()
    preds_top2 = preds_top2.cpu().detach().numpy()
    fpr, tpr, thresholds = roc_curve(labels, preds, pos_label=0)
    auc_value = auc(fpr, tpr)
    recall = recall_score(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    acc2 = top_k_accuracy_score(labels, preds_top2, k=2)
    acc3 = top_k_accuracy_score(labels, preds_top2, k=3)
    logger.info('RECALL:{:.4}, AUC: {:.6f}, ACC: {:.4f},ACC2: {:.4},ACC3: {:.4}'
                .format(recall,auc_value, acc, acc2, acc3))

    # write tensorboard

    dist.barrier()
    end_time = time.time()
    duration = end_time - start_time

    hour = duration // 3600
    minute = (duration % 3600) // 60
    sec = duration % 60
    logger.info('Testing cost %dh%dm%ds' % (hour, minute, sec))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', required=True, default='resnet', help='choose one backbone (s3dg or resnet)')

    parser.add_argument('--batch_size', type=int, default=4, help='batch size of per gpu, default is 4')
    parser.add_argument('--warmup_step', type=int, default=0, help='set the warmup step size, default is 0')
    parser.add_argument('--lr', type=int, default=0.0005, help='learning rate, default is 1e-4')
    parser.add_argument('--num_sample', type=int, default=1600, help='pairs num sampled from pair txt, default is 1600')
    parser.add_argument('--num_clip', type=int, default=16, help='frames num sampled from one video, default is 16')
    parser.add_argument('--num_workers', type=int, default=10, help='num_worker, default is 10')
    parser.add_argument('--save_epochs', type=int, default=100, help='save epochs frequency, default is 100')
    parser.add_argument('--max_epoch', type=int, default=300, help='the max epochs, default is 300')
    parser.add_argument('--seq_loss', type=int, default=1.0, help='the W of seq loss, default: 1')
    parser.add_argument('--info_loss', type=int, default=1.0, help='the W of seq loss, default: 1')
    parser.add_argument('--dist', type=str, default='NormL2', help='the distance of inference final scores ')

    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--unfreeze_backbone', dest='freeze_backbone', action='store_false')
    parser.set_defaults(freeze_backbone=True)

    parser.add_argument('--freeze_BN', action='store_true')
    parser.add_argument('--unfreeze_BN', dest='freeze_BN', action='store_false')
    parser.set_defaults(freeze_BN=False)

    parser.add_argument('--random_sample', action='store_true')
    parser.add_argument('--uniform_sample', dest='random_sample', action='store_false')
    parser.set_defaults(random_sample=True)

    parser.add_argument('--pre_load', action='store_true')
    parser.add_argument('--unpre_load', dest='pre_load', action='store_false')
    parser.set_defaults(pre_load=False)

    parser.add_argument('--pair', action='store_true')
    parser.add_argument('--unpair', dest='pair', action='store_false')
    parser.set_defaults(pair=True)

    parser.add_argument('--data_aug', action='store_true')
    parser.add_argument('--undata_aug', dest='data_aug', action='store_false')
    parser.set_defaults(data_aug=False)

    parser.add_argument('--multi_TE', action='store_true')
    parser.add_argument('--single_TE', dest='multi_TE', action='store_false')
    parser.set_defaults(multi_TE=False)

    # --------------------------------------------------------------
    # gumble softmax
    parser.add_argument('--use_gumbel', action='store_true')
    parser.add_argument('--unuse_gumbel', dest='use_gumbel',  action='store_false')
    parser.set_defaults(use_gumbel=False)

    parser.add_argument('--gt_type', type=str, default='sort', help='the type of gumbel loss gt type')
    # --------------------------------------------------------------
    parser.add_argument('--warmup_LR', action='store_true')
    parser.add_argument('--unwarmup_LR', dest='warmup_LR', action='store_false')
    parser.set_defaults(warmup_LR=False)


    # sim matrix
    parser.add_argument('--use_sim', action='store_true')
    parser.add_argument('--unuse_sim', dest='use_sim', action='store_false')
    parser.set_defaults(use_sim=False)

    parser.add_argument('--sim_dist',  default='l2', help='choose one sim distance (l2 or attn)')

    parser.add_argument('--concat_sim', action='store_true')
    parser.add_argument('--unconcat_sim', dest='concat_sim', action='store_false')
    parser.set_defaults(concat_sim=False)

    parser.add_argument('--add_sim', action='store_true')
    parser.add_argument('--unadd_sim', dest='add_sim', action='store_false')
    parser.set_defaults(add_sim=False)

    parser.add_argument('--sim_pos', default='first', help='set the pos of sim matrix')
    # --------------------------------------------------------------
    # language loss
    parser.add_argument('--use_text', action='store_true')
    parser.add_argument('--unuse_text', dest='use_text', action='store_false')
    parser.set_defaults(use_text=False)

    parser.add_argument('--use_neg_text', action='store_true')
    parser.add_argument('--unuse_neg_text', dest='use_neg_text', action='store_false')
    parser.set_defaults(use_neg_text=False)

    parser.add_argument('--info_mask', action='store_true')
    parser.add_argument('--uninfo_mask', dest='info_mask', action='store_false')
    parser.set_defaults(info_mask=True)

    parser.add_argument('--info_ddp', action='store_true')
    parser.add_argument('--uninfo_ddp', dest='info_ddp', action='store_false')
    parser.set_defaults(info_ddp=False)

    # --------------------------------------------------------------

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--uneval', dest='eval', action='store_false')
    parser.set_defaults(eval=True)

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--undebug', dest='debug', action='store_false')
    parser.set_defaults(debug=False)

    parser.add_argument('--use_ddp', action='store_true')
    parser.add_argument('--unuse_ddp', dest='use_ddp', action='store_false')
    parser.set_defaults(use_ddp=True)
    parser.add_argument("--local_rank", default=-1, type=int)

    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--unuse_amp', dest='use_amp', action='store_false')
    parser.set_defaults(use_amp=True)


    parser.add_argument('--cfg_from_args', action='store_true')
    parser.add_argument('--uncfg_from_args', dest='cfg_from_args', action='store_false')
    parser.set_defaults(cfg_from_args=False)


    parser.add_argument('--config', default='configs/train_config.yml', help='config file path')
    parser.add_argument('--eval_config', default='configs/eval_retrival_config.yml', help='config file path')
    parser.add_argument('--save_path', default=None, help='path to save models and log')
    parser.add_argument('--load_path', default=None, help='path to load the model')
    parser.add_argument('--log_name', default='train_log', help='log name')
    parser.add_argument('--tensorboard', required=False, default='default', help='tensorboard log name cannot be blank')

    args = parser.parse_args()
    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


    return cfg, eval_cfg


if __name__ == "__main__":

    ckpt_path = '/public/home/dongsx/svip/csv_logs/1109010_clip_csv_sample0.8k_lr5e-4_5e-6_bs8_ns16_wo_seq_wo_pair_info_gumbel_sort_wo_mask/1109010_clip_csv_sample0.8k_lr5e-4_5e-6_bs8_ns16_wo_seq_wo_pair_info_gumbel_sort_wo_mask/save_models/best_model.tar'
    cmd = r'CUDA_VISIBLE_DEVICES=0,1,2,3  python -m torch.distributed.launch --nproc_per_node=4  text_retrival.py --config configs/train_config.yml --unfreeze_backbone --backbone clip  --unpair --use_text --use_gumbel --gt_type sort --uninfo_mask'


    args = parse_args()
    args.load_path = ckpt_path
    cfg = get_cfg_defaults()
    if args.config:
        cfg.merge_from_file(args.config)
    if args.eval:
        eval_cfg = get_cfg_defaults()
        if args.eval_config:
            eval_cfg.merge_from_file(args.eval_config)

    else:
        raise IOError('need value')

    use_cuda = cfg.TRAIN.USE_CUDA and torch.cuda.is_available()

    eval_retrival(cfg, eval_cfg, args,)
