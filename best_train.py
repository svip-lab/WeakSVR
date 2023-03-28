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
    logger_path = os.path.join(cfg.TRAIN.SAVE_PATH, '/logs')
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
    # optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=0.01)


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

    if args.warmup_LR:
        w_lr = 1
        optimizer1 = torch.optim.AdamW(model.module.backbone.parameters(), lr=1e-7, weight_decay=0.01)
    else:
        w_lr = 1
        optimizer1 = torch.optim.AdamW(model.module.backbone.parameters(), lr=cfg.TRAIN.LR * 1e-2, weight_decay=0.01)

    # optimizer1 = torch.optim.AdamW(model.module.backbone.parameters(), lr=cfg.TRAIN.LR * 1e-2, weight_decay=0.01)
    # optimizer1 = torch.optim.AdamW(model.module.backbone.parameters(), lr=1e-7, weight_decay=0.01)
    optimizer2 = torch.optim.AdamW(finetune_params, lr=cfg.TRAIN.LR, weight_decay=0.01)
    if args.warmup_step == 0:

        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1,
                                                               T_max=cfg.TRAIN.MAX_EPOCH,
                                                               eta_min=cfg.TRAIN.LR * w_lr * 0.01)

        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2,
                                                               T_max=cfg.TRAIN.MAX_EPOCH,
                                                               eta_min=cfg.TRAIN.LR * 0.01)
    else:
        scheduler1 = CosineAnnealingWarmupRestarts(optimizer1,
                                                  first_cycle_steps=cfg.TRAIN.MAX_EPOCH,
                                                  cycle_mult=1.0,
                                                  max_lr=cfg.TRAIN.LR * 1e-2,
                                                  min_lr=cfg.TRAIN.LR * 0.01 * 1e-2,
                                                  warmup_steps=args.warmup_step,
                                                  gamma=1.0)

        scheduler2 = CosineAnnealingWarmupRestarts(optimizer2,
                                                  first_cycle_steps=cfg.TRAIN.MAX_EPOCH,
                                                  cycle_mult=1.0,
                                                  max_lr=cfg.TRAIN.LR,
                                                  min_lr=cfg.TRAIN.LR * 0.01,
                                                  warmup_steps=args.warmup_step,
                                                  gamma=1.0)
    # Load checkpoint
    start_epoch = 0
    if args.load_path and os.path.isfile(args.load_path):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(args.load_path, map_location=map_location)
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = 'module.' + k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
        # model.load_state_dict(checkpoint['model_state_dict'])
        optimizer1.load_state_dict(checkpoint['optimizer_state_dict1'])
        optimizer2.load_state_dict(checkpoint['optimizer_state_dict2'])
        start_epoch = checkpoint['epoch']
        logger.info('-> Loaded checkpoint %s (epoch: %d)' % (args.load_path, start_epoch))

    # Mulitple gpu
    # if torch.cuda.device_count() > 1 and torch.cuda.is_available():
    #     logger.info('Let us use %d GPUs' % torch.cuda.device_count())
    #     model = torch.nn.DataParallel(model)

    # Create checkpoint dir
    if cfg.TRAIN.SAVE_PATH:
        checkpoint_dir = os.path.join(cfg.TRAIN.SAVE_PATH, '/save_models')
        if not os.path.exists(checkpoint_dir) and dist.get_rank() == 0:
            os.makedirs(checkpoint_dir)
    else:
        checkpoint_dir = None
    # Start training
    Best_AUC_VAL = 0
    Best_AUC = 0
    train_loader = load_dataset_ddp(cfg, args, drop_last=True)
    test_loader = load_dataset_ddp(eval_cfg, args, drop_last=False)
    if cfg.DATASET.NAME =='COIN-SV' or cfg.DATASET.NAME =='DIVING48-SV':
        eval_cfg2 = eval_cfg
        eval_cfg2.DATASET.TXT_PATH = cfg.DATASET.TXT_PATH.replace('train_pairs.txt', 'val_pairs.txt')
        eval_cfg2.DATASET.MODE = 'val'
        val_loader = load_dataset_ddp(eval_cfg2, args, drop_last=False)

    if args.retrival:
        retrival_cfg = eval_cfg
        retrival_cfg.DATASET.TXT_PATH = cfg.DATASET.TXT_PATH.replace('train_pairs.txt', 'text_retrieval.txt')
        retrival_loader = load_dataset_retrival_ddp(retrival_cfg, args, drop_last=False)
    scaler = GradScaler()

    start_time = time.time()

    if dist.get_rank() == 0:
        t_epoch = tqdm(range(start_epoch, cfg.TRAIN.MAX_EPOCH))
    else:
        t_epoch = range(start_epoch, cfg.TRAIN.MAX_EPOCH)

    for epoch in t_epoch:
        if args.pair:
            w_cls = 1
        else:
            w_cls = 0
        train_loader.sampler.set_epoch(epoch)
        test_loader.sampler.set_epoch(epoch)
        loss_per_epoch, loss_per_epoch_cls, loss_per_epoch_seq, loss_per_epoch_info,loss_per_epoch_gumbel = 0, 0, 0, 0,0
        num_true_pred = 0
        logit_scale1 = 0
        logit_scale2 = 0
        # train one epoch
        model.train()
        if dist.get_rank() == 0 and False:
            iter_train = tqdm(train_loader)
        else:
            iter_train = train_loader
        for iter, sample in enumerate(iter_train):
            # break
            # if iter == 2 and args.debug:
            #     break
            if args.pair:
                frames1 = frames_preprocess(sample['clips1'][0], flipped=False).to(local_rank, non_blocking=True)
                frames2 = frames_preprocess(sample['clips2'][0], flipped=False).to(local_rank, non_blocking=True)

                labels1 = sample['labels1'].to(local_rank, non_blocking=True)
                labels2 = sample['labels2'].to(local_rank, non_blocking=True)

                label_token1 = sample['label_token1'].to(local_rank, non_blocking=True)
                label_token_phrase1 = sample['label_token_phrase1'][0].to(local_rank, non_blocking=True)
                label_phrase_num1 = sample['label_token_phrase1'][1].to(local_rank, non_blocking=True)

                label_token2 = sample['label_token2'].to(local_rank, non_blocking=True)
                label_token_phrase2 = sample['label_token_phrase2'][0].to(local_rank, non_blocking=True)
                label_phrase_num2 = sample['label_token_phrase2'][1].to(local_rank, non_blocking=True)
                label_neg_token1 = sample['label_neg_token1'].to(local_rank, non_blocking=True)
                label_neg_token2 = sample['label_neg_token2'].to(local_rank, non_blocking=True)
                if not args.freeze_backbone:
                    optimizer1.zero_grad()
                optimizer2.zero_grad()

                if args.use_amp:
                    with autocast():

                        pred1, text_feature1, seq_features1, embed1, logit_scale1_1, all_text_feature1, text_feature_phrase1, logit_scale1_2 = model(frames1,
                                                                                                             label_token1,
                                                                                                             label_token_phrase1,
                                                                                                             label_neg_token1)
                        pred2, text_feature2, seq_features2, embed2, logit_scale2_1, all_text_feature2, text_feature_phrase2, logit_scale2_2 = model(frames2,
                                                                                                             label_token2,
                                                                                                             label_token_phrase2,
                                                                                                             label_neg_token2)

                        loss_cls = compute_cls_loss(pred1, labels1) + compute_cls_loss(pred2, labels2)
                        if cfg.MODEL.SEQ_LOSS_COEF != 0:
                            loss_seq = compute_seq_loss(seq_features1, seq_features2)
                        else:
                            loss_seq = torch.zeros([1]).to(local_rank)
                        if args.use_text:
                            if args.use_neg_text:
                                loss_info1 = compute_info_loss_neg(embed1, text_feature1, all_text_feature1, logit_scale1_1,)
                                loss_info2 = compute_info_loss_neg(embed2, text_feature2, all_text_feature2, logit_scale2_1,)
                            else:
                                loss_info1 = compute_info_loss(embed1, text_feature1, labels1, logit_scale1_1)
                                loss_info2 = compute_info_loss(embed2, text_feature2, labels2, logit_scale2_1)
                            loss_info = loss_info1 + loss_info2
                        else:
                            loss_info = torch.zeros([1]).to(local_rank)

                        if args.use_gumbel:
                            loss_gumbel1 = compute_gumbel_loss(seq_features1, text_feature_phrase1, label_phrase_num1, logit_scale1_2,max_seq_length=MAX_SEQ_LENGTH, gt_type=args.gt_type,labels1=sample['labels1_raw'])
                            loss_gumbel2 = compute_gumbel_loss(seq_features2, text_feature_phrase2, label_phrase_num2, logit_scale1_2,max_seq_length=MAX_SEQ_LENGTH, gt_type=args.gt_type,labels1=sample['labels2_raw'])
                            loss_gumbel = loss_gumbel1 + loss_gumbel2
                        else:
                            loss_gumbel = torch.zeros([1]).to(local_rank)
                        loss = w_cls * loss_cls + cfg.MODEL.SEQ_LOSS_COEF * loss_seq + cfg.MODEL.INFO_LOSS_COEF * loss_info \
                               + cfg.MODEL.GUMBEL_LOSS_COEF * loss_gumbel

                # Update weights
                if args.use_amp:
                    scaler.scale(loss).backward()
                    if not args.freeze_backbone:
                        scaler.step(optimizer1)
                    scaler.step(optimizer2)
                    scaler.update()

                # else:
                #     loss.backward()
                #     optimizer.step()

                num_true_pred_per = (torch.argmax(pred1, dim=-1) == labels1).sum() + \
                                    (torch.argmax(pred2, dim=-1) == labels2).sum()
                model = update_logit_scale(model)
                torch.cuda.synchronize()

                # AUC and WDR
                num_true_pred += all_reduce_sum(num_true_pred_per)
                loss_per_epoch_cls += all_reduce_mean(loss_cls.item())
                loss_per_epoch_seq += all_reduce_mean(loss_seq.item())
                loss_per_epoch_info += all_reduce_mean(loss_info.item())
                loss_per_epoch_gumbel += all_reduce_mean(loss_gumbel.item())
                loss_per_epoch += all_reduce_mean(loss.item())
                logit_scale1 += logit_scale1_1.item()
                logit_scale2 += logit_scale1_2.item()
            # ------------------------------------------------------------------------
            else:
                # unuse pair data
                frames1 = frames_preprocess(sample['clips1'][0], flipped=False).to(local_rank, non_blocking=True)
                labels1 = sample['labels1'].to(local_rank, non_blocking=True)
                label_token1 = sample['label_token1'].to(local_rank, non_blocking=True)
                label_token_phrase1 = sample['label_token_phrase1'][0].to(local_rank, non_blocking=True)
                label_phrase_num1 = sample['label_token_phrase1'][1].to(local_rank, non_blocking=True)
                label_neg_token1 = sample['label_neg_token1'].to(local_rank, non_blocking=True)
                if not args.freeze_backbone:
                    optimizer1.zero_grad()
                optimizer2.zero_grad()

                if args.use_amp:
                    with autocast():
                        pred1, text_feature1, seq_features1, embed1, logit_scale1_1, all_text_feature1, text_feature_phrase1, logit_scale1_2 = model(
                            frames1,
                            label_token1,
                            label_token_phrase1,
                            label_neg_token1)
                        if not args.pair:
                            loss_cls = compute_cls_loss(pred1, labels1)
                        else:
                            loss_cls = compute_cls_loss(pred1, labels1)
                        # loss_cls = torch.zeros([1]).to(local_rank)

                        if args.use_text:
                            if args.use_neg_text:
                                loss_info = compute_info_loss_neg(embed1, text_feature1, all_text_feature1, logit_scale1_1,)
                            else:
                                loss_info = compute_info_loss(embed1, text_feature1, labels1, logit_scale1_1)
                        else:
                            loss_info = torch.zeros([1]).to(local_rank)
                        if args.use_gumbel:
                            loss_gumbel = compute_gumbel_loss(seq_features1, text_feature_phrase1, label_phrase_num1, logit_scale1_2, max_seq_length=MAX_SEQ_LENGTH, gt_type=args.gt_type)
                        else:
                            loss_gumbel = torch.zeros([1]).to(local_rank)
                        loss = w_cls * loss_cls + cfg.MODEL.INFO_LOSS_COEF * loss_info + cfg.MODEL.GUMBEL_LOSS_COEF * loss_gumbel
                        # print(loss)
                        # if torch.isnan(loss):
                        #     ipdb.set_trace()
                # else:
                #     pred1, seq_features1, embed1, logit_scale1 = model(frames1)
                #     loss_cls = compute_cls_loss(pred1, labels1)
                #     loss = 1 * loss_cls

                model = update_logit_scale(model)
                # AUC and WDR
                loss_seq = 0
                # Update weights
                if args.use_amp:
                    scaler.scale(loss).backward()
                    if not args.freeze_backbone:
                        scaler.step(optimizer1)
                    scaler.step(optimizer2)
                    scaler.update()
                # else:
                #     loss.backward()
                #     optimizer.step()

                num_true_pred_per = (torch.argmax(pred1, dim=-1) == labels1).sum()
                torch.cuda.synchronize()

                # AUC and WDR
                num_true_pred += all_reduce_sum(num_true_pred_per)
                loss_per_epoch_cls += all_reduce_mean(loss_cls.item())
                loss_per_epoch_seq += 0
                loss_per_epoch_info += all_reduce_mean(loss_info.item())
                loss_per_epoch_gumbel += all_reduce_mean(loss_gumbel.item())
                loss_per_epoch += all_reduce_mean(loss.item())
                logit_scale1 += logit_scale1_1.item()
                logit_scale2 += logit_scale1_2.item()
        # Log training statistics
        loss_per_epoch /= (iter + 1)
        loss_per_epoch_cls /= (iter + 1)
        loss_per_epoch_seq /= (iter + 1)
        loss_per_epoch_info /= (iter + 1)
        loss_per_epoch_gumbel /= (iter + 1)
        logit_scale1 /= (iter+1)
        logit_scale2 /= (iter+1)
        if args.pair:
            accuracy = num_true_pred / (cfg.DATASET.NUM_SAMPLE * 2)
        else:
            accuracy = num_true_pred / cfg.DATASET.NUM_SAMPLE
        logger.info('Epoch [{}/{}], LR1: {:.6f}, LR2: {:.6f}, Accuracy: {:.4f}, Loss: {:.4f}, '
                    'Loss_cls: {:.4f}, loss_seq:{:.4f}, loss_info:{:.4f}, loss_gumble:{:.4f}'
                    .format(epoch, cfg.TRAIN.MAX_EPOCH, optimizer1.param_groups[0]['lr'], optimizer2.param_groups[0]['lr'],
                            accuracy,
                            loss_per_epoch,
                            loss_per_epoch_cls, loss_per_epoch_seq, loss_per_epoch_info, loss_per_epoch_gumbel))
        if dist.get_rank() == 0:
            writer.add_scalar('accuracy/train', accuracy, epoch)
        # eval
        model.eval()
        # -------------------------------------------
        # test on test set
        auc_value, wdr_value = eval_per_epoch(model, test_loader, local_rank, eval_cfg, args)
        # with torch.no_grad():
        #     labels, preds, labels1_all, labels2_all = None, None, None, None
        #     for iter, sample in enumerate(test_loader):
        #         if iter == 3 and args.debug:
        #             break
        #         frames1_list = sample['clips1']
        #         frames2_list = sample['clips2']
        #         assert len(frames1_list) == len(frames2_list), 'frames1_list:{},frames2_list{}'.format(
        #             len(frames1_list), len(frames2_list))
        #
        #         labels1 = sample['labels1']
        #         labels2 = sample['labels2']
        #         label = torch.tensor(np.array(labels1) == np.array(labels2)).to(local_rank)
        #
        #         embeds1_list = []
        #         embeds2_list = []
        #
        #         for i in range(len(frames1_list)):
        #             frames1 = frames_preprocess(frames1_list[i]).to(local_rank, non_blocking=True)
        #             frames2 = frames_preprocess(frames2_list[i]).to(local_rank, non_blocking=True)
        #             embeds1 = model(frames1, embed=True)
        #             embeds2 = model(frames2, embed=True)
        #
        #             embeds1_list.append(embeds1.unsqueeze(dim=0))
        #             embeds2_list.append(embeds2.unsqueeze(dim=0))
        #
        #         embeds1_avg = (torch.cat(embeds1_list, dim=0)).mean(dim=0)
        #         embeds2_avg = (torch.cat(embeds2_list, dim=0)).mean(dim=0)
        #         pred = pred_dist(args.dist, embeds1_avg, embeds2_avg)
        #
        #         torch.cuda.synchronize()
        #
        #         # gather from other gpu
        #         pred = all_gather_concat(pred)
        #         label = all_gather_concat(label)
        #         labels1 = all_gather_object(labels1)
        #         labels2 = all_gather_object(labels2)
        #
        #         # add all data to list
        #         if iter == 0:
        #             preds = pred
        #             labels = label
        #             labels1_all = labels1
        #             labels2_all = labels2
        #         else:
        #             preds = torch.cat([preds, pred])
        #             labels = torch.cat([labels, label])
        #             labels1_all += labels1
        #             labels2_all += labels2
        #
        # fpr, tpr, thresholds = roc_curve(labels.cpu().detach().numpy(), preds.cpu().detach().numpy(), pos_label=0)
        # auc_value = auc(fpr, tpr)
        # wdr_value = compute_WDR(preds, labels1_all, labels2_all, eval_cfg.DATASET.NAME)
        logger.info('Epoch [{}/{}], AUC: {:.6f}, WDR: {:.4f}.'
                    .format(epoch, cfg.TRAIN.MAX_EPOCH, auc_value, wdr_value))

        if auc_value > Best_AUC:
            Best_AUC = auc_value
            save_new_best_ckpt = True
        else:
            save_new_best_ckpt = False
        # -------------------------------------------
        # -------------------------------------------
        # test on valid set
        if (cfg.DATASET.NAME =='COIN-SV' or cfg.DATASET.NAME =='DIVING48-SV') and False:
            val_loader.sampler.set_epoch(epoch)
            auc_value_val, wdr_value_val = eval_per_epoch(model, val_loader, local_rank, eval_cfg2, args)
            logger.info('Epoch [{}/{}], VAL: AUC: {:.6f}, WDR: {:.4f}.'
                        .format(epoch, cfg.TRAIN.MAX_EPOCH, auc_value_val, wdr_value_val))
            if auc_value_val > Best_AUC_VAL:
                Best_AUC_VAL = auc_value_val
                save_new_val_best_ckpt = True
            else:
                save_new_val_best_ckpt = False
        else:
            auc_value_val = 0
            wdr_value_val = 0
            save_new_val_best_ckpt = False
        # -------------------------------------------

        if args.retrival:
            retrieval_auc = retrieval(model, retrival_loader, args, logger, epoch)
            if dist.get_rank() == 0:
                writer.add_scalar('AUC/retrieval', retrieval_auc, epoch)
        # -------------------------------------------
        # write tensorboard
        if dist.get_rank() == 0:
            writer.add_scalar('AUC/val', auc_value_val, epoch)
            writer.add_scalar('WDR/val', wdr_value_val, epoch)
            writer.add_scalar('AUC/test', auc_value, epoch)
            writer.add_scalar('WDR/test', wdr_value, epoch)
            writer.add_scalar('learning_rate/backbone', optimizer1.param_groups[0]['lr'], epoch)
            writer.add_scalar('learning_rate/other', optimizer2.param_groups[0]['lr'], epoch)
            writer.add_scalar('logit_scale/logit_scale1', logit_scale1, epoch)
            writer.add_scalar('logit_scale/logit_scale2', logit_scale2, epoch)
            writer.add_scalar('Loss/tr_total', loss_per_epoch, epoch)
            writer.add_scalar('Loss/tr_cls', loss_per_epoch_cls, epoch)
            writer.add_scalar('Loss/tr_sep', loss_per_epoch_seq, epoch)
            writer.add_scalar('Loss/tr_info', loss_per_epoch_info, epoch)
            writer.add_scalar('Loss/tr_gumbel', loss_per_epoch_gumbel, epoch)
            writer.close()

        # Save model every X epochs
        if dist.get_rank() == 0 and (save_new_best_ckpt or save_new_val_best_ckpt) and args.save_model:
            save_dict = {'epoch': epoch,  # after training one epoch, the start_epoch should be epoch+1
                         'optimizer_state_dict1': optimizer1.state_dict(),
                         'optimizer_state_dict2': optimizer2.state_dict(),
                         'auc_value_val': auc_value_val,
                         'wdr_value_val': wdr_value_val,
                         'auc_value': auc_value,
                         'wdr_value': wdr_value,
                         'loss': loss.item(),
                         }
            try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
                save_dict['model_state_dict'] = model.module.state_dict()
            except:
                save_dict['model_state_dict'] = model.state_dict()
            if save_new_val_best_ckpt:
                save_name = 'best_val_model' + '.tar'
                torch.save(save_dict, os.path.join(checkpoint_dir, save_name))
                logger.info('Save ' + os.path.join(checkpoint_dir, save_name) + ' done!')
            if save_new_best_ckpt:
                save_name = 'best_model' + '.tar'
                torch.save(save_dict, os.path.join(checkpoint_dir, save_name))
                logger.info('Save ' + os.path.join(checkpoint_dir, save_name) + ' done!')
        # if dist.get_rank() == 0:
        #     save_dict = {'epoch': epoch,  # after training one epoch, the start_epoch should be epoch+1
        #                  'optimizer_state_dict1': optimizer1.state_dict(),
        #                  'optimizer_state_dict2': optimizer2.state_dict(),
        #                  'auc_value_val': auc_value_val,
        #                  'wdr_value_val': wdr_value_val,
        #                  'auc_value': auc_value,
        #                  'wdr_value': wdr_value,
        #                  'loss': loss.item(),
        #                  }
        #     try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
        #         save_dict['model_state_dict'] = model.module.state_dict()
        #     except:
        #         save_dict['model_state_dict'] = model.state_dict()
        #     save_name = str(epoch) + '.tar'
        #     torch.save(save_dict, os.path.join(checkpoint_dir, save_name))
        #     logger.info('Save ' + os.path.join(checkpoint_dir, save_name) + ' done!')
        dist.barrier()
        # Learning rate decay
        scheduler1.step()
        scheduler2.step()

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
        for iter, sample in enumerate(val_loader):
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
