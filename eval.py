from utils.load_parse_args import parse_args
import os
import random
import time
import warnings
import yaml
import ipdb
import numpy as np
import torch
import torch.distributed as dist
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from sklearn.metrics import auc, roc_curve
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm
import torch.nn as nn
from Data.clip_dataset import load_dataset_ddp
from configs.defaults import get_cfg_defaults
from models.build_model import build_model,update_logit_scale
from utils.logger import setup_logger
from utils.loss import compute_cls_loss, compute_seq_loss, compute_info_loss_neg, compute_gumbel_loss
from collections import OrderedDict
from utils.metrics import compute_WDR, pred_dist
from utils.preprocess import frames_preprocess
from utils.utils_distributed import all_gather_concat,all_reduce_mean,all_reduce_sum,all_gather_object
from retrival import retrieval
from Data.clip_dataset_retrival import load_dataset_retrival_ddp
warnings.filterwarnings("ignore")

# TORCH_DISTRIBUTED_DEBUG = 'Detail'



# torch.autograd.set_detect_anomaly(True)
def setup(local_rank):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')


def check_args_cfg(args, cfg):
    pass
def update_yaml():
    yamlpath = os.path.join('./train_yaml', 'test.yaml')

def init_log(cfg=None, eval_cfg=None, args=None, local_rank=0,):

    cfg, eval_cfg = update_cfg_from_args(cfg, eval_cfg, args)
    logger_path = os.path.join(cfg.TRAIN.SAVE_PATH, args.tensorboard + '/logs')
    logger = setup_logger('Sequence Verification', logger_path, args.log_name, args.local_rank)
    logger.info('----------------Running with args----------------\n{}\n'.format(vars(args)))
    if args.cfg_from_args:
        logger.info('-------------Update training cfg from args----------------\n')
    logger.info('Running training with config:\n{}\n'.format(cfg))
    if args.eval:
        logger.info('-------------Update eval cfg from train config-------------\n')
        logger.info('Running eval with config:\n{}\n'.format(eval_cfg))


    # if cfg.MODEL.SAVE_MODEL_LOG and args.local_rank==0:
    #     model = build_model(cfg=cfg, args=args, model_log=True).to(args.local_rank)
    #     if args.backbone == 'resnet':
    #         model_log = summary(model, (16, 3, 180, 320), depth=3)
    #     else:
    #         model_log = summary(model, (16, 3, 224, 224), depth=3)
    #     logger.info('Running training with model:\n{}\n'.format(model_log))
    #     del model

    return logger


def train(cfg, eval_cfg, args,):
    if cfg.DATASET.NAME == 'CSV':
        MAX_SEQ_LENGTH = 20
    elif cfg.DATASET.NAME == 'COIN-SV':
        MAX_SEQ_LENGTH = 25
    elif cfg.DATASET.NAME == 'DIVING48-SV':
        MAX_SEQ_LENGTH = 4
    else:
        raise ValueError('wrong cfg,DATASET.NAME')
    local_rank = args.local_rank
    setup(local_rank)
    setup_seed(cfg.TRAIN.SEED + local_rank)
    logger = init_log(cfg, eval_cfg, args, local_rank)
    if dist.get_rank() == 0:
        log_dir = args.tensorboard
        writer = SummaryWriter(log_dir=os.path.join('log/', log_dir))
    else:
        writer = None

    model = build_model(cfg=cfg, args=args, model_log=False).to(local_rank)


    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    backbone_params = list(map(id, model.module.backbone.parameters()))
    finetune_params = filter(lambda p: id(p) not in backbone_params, model.parameters())

    # optimizer = torch.optim.AdamW([
    #                 {'params': model.module.backbone.parameters(), 'lr': 1e-7},
    #                 {'params': finetune_params},
    #             ], lr=cfg.TRAIN.LR, weight_decay=0.01)

    if not args.pair:
        args.info_mask = False

    if args.info_ddp:
        if args.info_mask:
            from utils.loss import compute_info_loss_mask_ddp as compute_info_loss
            logger.info('info w mask, w ddp')
        else:
            from utils.loss import compute_info_loss_ddp as compute_info_loss
            logger.info('info w/o mask, w ddp')
    else:
        if args.info_mask:
            from utils.loss import compute_info_loss_mask as compute_info_loss
            logger.info('info w mask, w/o ddp')
        else:
            from utils.loss import compute_info_loss as compute_info_loss
            logger.info('info w/0 mask, w/o ddp')

    # Load checkpoint
    start_epoch = 0
    if args.load_path and os.path.isfile(args.load_path):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(args.load_path, map_location=map_location)
        new_state_dict = OrderedDict()
        import ipdb
        # ipdb.set_trace()
        for k, v in checkpoint['model_state_dict'].items():
            name = 'module.' + k
            new_state_dict[name] = v
        # ipdb.set_trace()

        model.load_state_dict(new_state_dict, strict=False)
        # model.load_state_dict(checkpoint['model_state_dict'])

        start_epoch = checkpoint['epoch']
        logger.info('-> Loaded checkpoint %s (epoch: %d)' % (args.load_path, start_epoch))

    # Create checkpoint dir
    if cfg.TRAIN.SAVE_PATH:
        checkpoint_dir = os.path.join(cfg.TRAIN.SAVE_PATH, args.tensorboard + '/save_models')
        if not os.path.exists(checkpoint_dir) and dist.get_rank() == 0:
            os.makedirs(checkpoint_dir)
    else:
        checkpoint_dir = None

    test_loader = load_dataset_ddp(eval_cfg, args, drop_last=False)
    # if cfg.DATASET.NAME =='COIN-SV' or cfg.DATASET.NAME =='DIVING48-SV':
    #     eval_cfg2 = eval_cfg
    #     eval_cfg2.DATASET.TXT_PATH = cfg.DATASET.TXT_PATH.replace('train_pairs.txt', 'val_pairs.txt')
    #     eval_cfg2.DATASET.MODE = 'val'
    #     val_loader = load_dataset_ddp(eval_cfg2, args, drop_last=False)

    if args.retrival:
        retrival_cfg = eval_cfg
        retrival_cfg.DATASET.TXT_PATH = cfg.DATASET.TXT_PATH.replace('train_pairs.txt', 'text_retrieval.txt')
        retrival_loader = load_dataset_retrival_ddp(retrival_cfg, args, drop_last=False)


    start_time = time.time()


    # eval
    model.eval()
    # -------------------------------------------
    # test on test set
    auc_value, wdr_value = eval_per_epoch(model, test_loader, local_rank, eval_cfg, args)
    logger.info('Epoch [{}/{}], AUC: {:.6f}, WDR: {:.4f}.'
                .format(start_epoch, cfg.TRAIN.MAX_EPOCH, auc_value, wdr_value))

    # -------------------------------------------
    dist.barrier()
    end_time = time.time()
    duration = end_time - start_time

    hour = duration // 3600
    minute = (duration % 3600) // 60
    sec = duration % 60
    logger.info('Training cost %dh%dm%ds' % (hour, minute, sec))

def eval_per_epoch(model, val_loader,local_rank,eval_cfg,args):
    with torch.no_grad():
        labels, preds, labels1_all, labels2_all = None, None, None, None
        for iter, sample in tqdm(enumerate(val_loader)):
            # if iter == 1 and args.debug:
            if iter == 1 and args.debug:
                break
            frames1_list = sample['clips1']
            frames2_list = sample['clips2']
            assert len(frames1_list) == len(frames2_list), 'frames1_list:{},frames2_list{}'.format(
                len(frames1_list), len(frames2_list))

            labels1 = sample['labels1']
            labels2 = sample['labels2']
            label = torch.tensor(np.array(labels1) == np.array(labels2)).to(local_rank)

            embeds1_list = []
            embeds2_list = []

            for i in range(len(frames1_list)):
                frames1 = frames_preprocess(frames1_list[i]).to(local_rank, non_blocking=True)
                frames2 = frames_preprocess(frames2_list[i]).to(local_rank, non_blocking=True)
                embeds1 = model(frames1, embed=True)
                embeds2 = model(frames2, embed=True)

                embeds1_list.append(embeds1.unsqueeze(dim=0))
                embeds2_list.append(embeds2.unsqueeze(dim=0))

            embeds1_avg = (torch.cat(embeds1_list, dim=0)).mean(dim=0)
            embeds2_avg = (torch.cat(embeds2_list, dim=0)).mean(dim=0)
            pred = pred_dist(args.dist, embeds1_avg, embeds2_avg)

            torch.cuda.synchronize()

            # gather from other gpu
            pred = all_gather_concat(pred)
            label = all_gather_concat(label)
            labels1 = all_gather_object(labels1)
            labels2 = all_gather_object(labels2)

            # add all data to list
            if iter == 0:
                preds = pred
                labels = label
                labels1_all = labels1
                labels2_all = labels2
            else:
                preds = torch.cat([preds, pred])
                labels = torch.cat([labels, label])
                labels1_all += labels1
                labels2_all += labels2

    fpr, tpr, thresholds = roc_curve(labels.cpu().detach().numpy(), preds.cpu().detach().numpy(), pos_label=0)
    auc_value_val = auc(fpr, tpr)
    wdr_value_val = compute_WDR(preds, labels1_all, labels2_all, eval_cfg.DATASET.NAME)
    return auc_value_val, wdr_value_val


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def update_cfg_from_args(cfg, eval_cfg, args):

    if args.cfg_from_args:
        cfg.TRAIN.BATCH_SIZE = args.batch_size
        cfg.TRAIN.LR = args.lr
        cfg.DATASET.NUM_SAMPLE = args.num_samples
        cfg.MODEL.SEQ_LOSS_COEF = args.seq_loss
        cfg.MODEL.INFO_LOSS_COEF = args.info_loss
        cfg.MODEL.SAVE_EPOCHS = args.save_epochs
        cfg.TRAIN.MAX_EPOCH = args.max_epoch
        cfg.DATASET.NUM_CLIP = args.num_clip
        cfg.DATASET.NUM_SAMPLE = args.NUM_SAMPLE

    cfg.TRAIN.BATCH_SIZE = cfg.TRAIN.BATCH_SIZE if args.pair else cfg.TRAIN.BATCH_SIZE*2
    # print(cfg.TRAIN.BATCH_SIZE)
    # cfg.DATASET.NUM_SAMPLE = cfg.DATASET.NUM_SAMPLE if args.pair else 800
    cfg.DATASET.RANDOM_SAMPLE = args.random_sample
    cfg.TRAIN.SAVE_PATH = os.path.join(cfg.TRAIN.SAVE_PATH, args.tensorboard)

    if args.eval:
        eval_cfg.TRAIN.BATCH_SIZE = cfg.TRAIN.BATCH_SIZE
        eval_cfg.DATASET.NUM_CLIP = cfg.DATASET.NUM_CLIP
        eval_cfg.DATASET.NUM_WORKERS = cfg.DATASET.NUM_WORKERS
        eval_cfg.DATASET.NAME = cfg.DATASET.NAME
        eval_cfg.DATASET.TXT_PATH = cfg.DATASET.TXT_PATH.replace('train_pairs.txt', 'test_pairs.txt')
        eval_cfg.DATASET.NUM_CLASS = cfg.DATASET.NUM_CLASS


    return cfg, eval_cfg


if __name__ == "__main__":

    args = parse_args()
    cfg = get_cfg_defaults()
    if args.config:
        cfg.merge_from_file(args.config)
    if args.eval:
        eval_cfg = get_cfg_defaults()
        if args.eval_config:
            eval_cfg.merge_from_file(args.eval_config)

    else:
        raise IOError('need value')
    print('Warning: IF USE PAIR DATA, PLS CHECK NUM CLIP.')
    use_cuda = cfg.TRAIN.USE_CUDA and torch.cuda.is_available()

    train(cfg, eval_cfg, args,)
