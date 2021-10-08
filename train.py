import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.distributed as dist

from net.MobileNetRefine.LWANet import lwa
from net.MobileNetRefine.mobilenet import mbv2
from net.MobileNetRefine.resnet import rf_lw50
# from net.TernausNet.ternausnet import UNet16 as TUnet
from net.TernausNet.tunet import UNet16 as TUnet
from net.Ours.base import TemporalNet
# from net.BiseNet.r18 import BiSeNet
from net.unet.unet import UNet
from net.Ours.DenseST import DenseST
from net.Ours.SpNet import spnet
from net.Ours.GlobalDenseST import GDST
####data

from dataset.Endovis2017 import endovis2017
from utils.losses import BCELoss
# from utils.EndoLoss import LossMulti
from utils.EndoLoss import LossMulti
from utils.metrics import compute_dice, compute_iou
from utils.summary import create_summary, create_logger, create_saver, DisablePrint
from utils.LoadModel import load_model
# from net.BiseNet.seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d

# Training settings
parser = argparse.ArgumentParser(description='real-time segmentation')

parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--dist', action='store_true')

parser.add_argument('--root_dir',
                    type=str,
                    default='/raid/wjc/logs/RealtimeSegmentation')
parser.add_argument('--dataset', type=str)
parser.add_argument('--data_tag', type=str, choices=['part', 'type'])
parser.add_argument('--log_name', type=str)

parser.add_argument('--arch',
                    type=str,
                    choices=[
                        'mb_rf', 'lwa', 'bise', 'unet', 'tunet', 'tpnet',
                        'spnet', 'densest', 'gdst', 'res50_rf'
                    ])
parser.add_argument('--load_model', type=str, default=None)

parser.add_argument('--folds', type=str)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--loss', type=str)

parser.add_argument('--gpus', type=str)

parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--val_interval', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=3)

parser.add_argument('--lstm',
                    type=str,
                    choices=['convlstm', 'btnlstm', 'grouplstm', 'kdlstm'])
parser.add_argument('--t', type=int)

parser.add_argument('--freeze_name', type=str)
parser.add_argument('--spatial_layer', type=int)
parser.add_argument('--global_n', type=int)
parser.add_argument('--need_pretrain', type=int)
parser.add_argument('--pre_name', type=str)
parser.add_argument('--pretrain_ep', type=int, default=20)
parser.add_argument('--decay', type=int, default=2)
parser.add_argument('--fusion_type', type=str)

parser.add_argument('--reset', action='store_true')
parser.add_argument('--reset_epoch', type=int)

cfg = parser.parse_args()
# os.chdir(cfg.root_dir)
cfg.folds = list(map(int, cfg.folds.split(',')))
# loss_functions = {'dice': DiceLoss(ignore_index=4), 'bce': BCELoss()}
loss_functions = {'bce': BCELoss()}
# rate = 1 if cfg.arch=='bise' else 1
rate = 1


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpus
    #     torch.manual_seed(317)
    torch.backends.cudnn.benchmark = True  # disable this if OOM at beginning of training
    num_gpus = torch.cuda.device_count()

    if cfg.dist:
        cfg.device = torch.device('cuda:%d' % cfg.local_rank)
        torch.cuda.set_device(cfg.local_rank)
        dist.init_process_group(backend='nccl',
                                init_method='env://',
                                world_size=num_gpus,
                                rank=cfg.local_rank)
    else:
        cfg.device = torch.device('cuda')

    for fold in cfg.folds:
        cfg.log_dir = os.path.join(cfg.root_dir, cfg.dataset, cfg.data_tag,
                                   cfg.log_name, 'logs', 'fold{}'.format(fold))
        cfg.ckpt_dir = os.path.join(cfg.root_dir, cfg.dataset, cfg.data_tag,
                                    cfg.log_name, 'ckpt',
                                    'fold{}'.format(fold))
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.ckpt_dir, exist_ok=True)
        saver = create_saver(cfg.local_rank, save_dir=cfg.ckpt_dir)
        logger = create_logger(cfg.local_rank, save_dir=cfg.log_dir)
        summary_writer = create_summary(cfg.local_rank, log_dir=cfg.log_dir)
        print = logger.info
        print(cfg)
        print('Setting up data...')

        if cfg.dataset == 'endovis2017':
            train_dataset = endovis2017('train',
                                        t=cfg.t,
                                        fold=fold,
                                        rate=rate,
                                        tag=cfg.data_tag,
                                        global_n=cfg.global_n)
            val_dataset = endovis2017('val',
                                      t=cfg.t,
                                      fold=fold,
                                      rate=rate,
                                      tag=cfg.data_tag,
                                      global_n=cfg.global_n)
            classes = train_dataset.class_num

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=cfg.num_workers,
                                                 pin_memory=True,
                                                 drop_last=False)
        compute_loss = loss_functions[cfg.loss]
        # build model
        if 'mb_rf' in cfg.arch:
            model = mbv2(classes + 1, imagenet=True, rate=1)
        elif 'res50_rf' in cfg.arch:
            model = rf_lw50(classes + 1, imagenet=True)
        elif 'lwa' in cfg.arch:
            model = lwa(classes + 1, imagenet=True)
        elif 'tunet' in cfg.arch:
            model = TUnet(in_channels=64,
                          num_classes=classes + 1,
                          pretrained=True)
#             compute_loss = LossMulti(jaccard_weight=1,num_classes=classes+1)
        elif 'unet' in cfg.arch:
            model = UNet(3, classes + 1)
        elif 'tpnet' in cfg.arch:
            assert (cfg.t > 1)
            model = TemporalNet(classes + 1,
                                batch_size=cfg.batch_size,
                                tag=cfg.lstm,
                                group=1)
        elif 'spnet' in cfg.arch:
            assert (cfg.global_n > 1)
            model = spnet(classes + 1,
                          imagenet=True,
                          global_n=cfg.global_n,
                          spatial_layer=cfg.spatial_layer)
        elif 'densest' in cfg.arch:
            assert (cfg.t > 1)
            model = DenseST(classes + 1, tag=cfg.lstm)
        elif 'gdst' in cfg.arch:
            assert (cfg.t > 1 and cfg.global_n > 0
                    and cfg.fusion_type is not None)
            model = GDST(classes + 1,
                         batch_size=cfg.batch_size,
                         tag=cfg.lstm,
                         group=1,
                         t=cfg.t,
                         global_n=cfg.global_n,
                         fusion_type=cfg.fusion_type)
        else:
            raise NotImplementedError

        optimizer = torch.optim.Adam(model.parameters(), cfg.lr)

        torch.cuda.empty_cache()

        def train(epoch):
            print('\n Epoch: %d' % epoch)
            model.train()
            tic = time.perf_counter()
            for batch_idx, batch in enumerate(train_loader):
                for k in batch:
                    if not k == 'path':
                        batch[k] = batch[k].to(device=cfg.device,
                                               non_blocking=True).float()
                outputs = model(batch['image'])
                if 'bise' in cfg.arch:
                    loss = compute_loss(outputs, batch['label'])
                elif cfg.arch in ['unet', 'tunet']:
                    loss = compute_loss(outputs, batch['label'])
                elif cfg.arch in [
                        'mb_rf', 'lwa', 'tpnet', 'densest', 'spnet', 'gdst',
                        'res50_rf'
                ]:
                    outputs = F.interpolate(outputs, scale_factor=4 // rate)
                    loss = compute_loss(outputs, batch['label'])
                else:
                    raise NotImplementedError
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch_idx % cfg.log_interval == 0:
                    duration = time.perf_counter() - tic
                    tic = time.perf_counter()
                    print(
                        '[%d/%d-%d/%d]' %
                        (epoch, cfg.num_epochs, batch_idx, len(train_loader)) +
                        'Dice_loss:{:.4f} Time:{:.4f}'.format(
                            loss.item(), duration))

                step = len(train_loader) * epoch + batch_idx
                summary_writer.add_scalar('loss/AVG', loss.item(), step)
            return

        def val_map(epoch):
            print('\n Val@Epoch: %d' % epoch)
            model.eval()
            torch.cuda.empty_cache()
            dices = []
            ious = []
            metrics = np.zeros((2, classes))
            with torch.no_grad():

                for inputs in val_loader:
                    inputs['image'] = inputs['image'].to(cfg.device).float()

                    tic = time.perf_counter()
                    output = model(inputs['image'])
                    if 'bise' in cfg.arch:
                        output = F.softmax(output, dim=1).cpu().numpy()
                    elif cfg.arch in [
                            'mb_rf', 'lwa', 'tpnet', 'densest', 'spnet',
                            'gdst', 'res50_rf'
                    ]:
                        output = F.interpolate(output, scale_factor=4 // rate)
                        output = F.softmax(output, dim=1)
                        output = F.one_hot(torch.argmax(output, dim=1),
                                           num_classes=classes + 1).permute(
                                               0, 3, 1, 2)
                        output = output.cpu().numpy()
                    elif cfg.arch in ['unet', 'tunet']:
                        output = F.softmax(output, dim=1).cpu().numpy()


#                         output = output.cpu().numpy()
                    else:

                        raise NotImplementedError
                    duration = time.perf_counter() - tic
                    dice = compute_dice(output,
                                        inputs['label'].numpy(),
                                        return_all=True)
                    iou = compute_iou(output,
                                      inputs['label'].numpy(),
                                      return_all=True)
                    dices.append(dice)
                    ious.append(iou)
                dices = np.array(dices)
                ious = np.array(ious)
                for i in range(classes):
                    metrics[0, i] = np.mean(dices[:, i][dices[:, i] > -1])
                    metrics[1, i] = np.mean(ious[:, i][ious[:, i] > -1])
                print(metrics)
                dc, jc = [
                    np.mean(metrics[i][metrics[i] > -1]) for i in range(2)
                ]
            print('Dice:{:.4f} IoU:{:.4f} Time:{:.4f}'.format(
                dc, jc, duration))
            summary_writer.add_scalar('Dice/Fold{}'.format(fold), dc, epoch)
            summary_writer.add_scalar('IoU/Fold{}'.format(fold), jc, epoch)
            return dc

        print('Starting training...')
        best = 0
        best_ep = 0
        model = model.to(cfg.device)

        if cfg.arch in ['densest', 'gdst', 'bgdst']:
            mem_path = os.path.join(cfg.root_dir, cfg.dataset, cfg.data_tag,
                                    'spnet', 'ckpt', 'fold{}'.format(fold),
                                    'checkpoint.t7')
            cfg.load_model = os.path.join(cfg.root_dir, cfg.dataset,
                                          cfg.data_tag, cfg.pre_name, 'ckpt',
                                          'fold{}'.format(fold),
                                          'checkpoint.t7')
            model = load_model(model, mem_path, False)
            model = load_model(model, cfg.load_model, False)
            model.encoder = load_model(model.encoder, cfg.load_model, False)
            model.decoder = load_model(model.decoder, cfg.load_model, False)

        if cfg.arch in ['tpnet', 'spnet', 'densest']:
            if cfg.need_pretrain:
                if cfg.freeze_name is None:
                    if cfg.arch == 'tpnet':
                        cfg.freeze_name = ['lstm']
                    elif cfg.arch == 'spnet':
                        cfg.freeze_name = ['memory']
                    else:
                        cfg.freeze_name = ['non_local']
                else:
                    cfg.freeze_name = cfg.freeze_name.split(',')
                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=cfg.batch_size * 2,
                    shuffle=True,
                    num_workers=cfg.num_workers,
                    pin_memory=True,
                    drop_last=True)
                for name, param in model.named_parameters():
                    if not name.split('.')[0] in cfg.freeze_name:
                        param.requires_grad = False
                    else:
                        print('{} NOT Freeze'.format(name))

                cfg.load_model = os.path.join(cfg.root_dir, cfg.dataset,
                                              cfg.data_tag, cfg.pre_name,
                                              'ckpt', 'fold{}'.format(fold),
                                              'checkpoint.t7')
                assert os.path.exists(cfg.load_model)
                if cfg.arch in ['tpnet', 'spnet']:
                    model.encoder = load_model(model.encoder, cfg.load_model)
                    model.decoder = load_model(model.decoder, cfg.load_model)
                print('Pretrain for {} epochs and save the best weight'.format(
                    cfg.pretrain_ep))
                for epoch in range(1, cfg.pretrain_ep + 1):
                    train(epoch)
                    save_map = val_map(epoch)
                    if save_map > best:
                        best = save_map
                        print(saver.save(model.state_dict(), 'stage1'))

                print(
                    'Finished Pretraining, reduce lr to a half and load the best weight'
                )
                optimizer = torch.optim.Adam(model.parameters(),
                                             cfg.lr / cfg.decay)
                cfg.load_model = os.path.join(cfg.ckpt_dir, 'stage1.t7')
                model = load_model(model, cfg.load_model)
                for name, param in model.named_parameters():
                    param.requires_grad = True
                best_ep = cfg.pretrain_ep
                best = 0

            else:
                cfg.load_model = os.path.join(cfg.ckpt_dir, 'stage1.t7')
                assert os.path.exists(cfg.load_model)
                model = load_model(model, cfg.load_model)
                optimizer = torch.optim.Adam(model.parameters(),
                                             cfg.lr / cfg.decay)
                best_ep = cfg.pretrain_ep
                best = 0

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=cfg.batch_size,
                                                   shuffle=True,
                                                   num_workers=cfg.num_workers,
                                                   pin_memory=True,
                                                   drop_last=True)
        if cfg.reset:
            cfg.reset_path = os.path.join(cfg.root_dir, cfg.dataset,
                                          cfg.data_tag, 'res50_rf', 'ckpt',
                                          'fold{}'.format(fold),
                                          'checkpoint.t7')
            model = load_model(model, cfg.reset_path)
            best = val_map(cfg.reset_epoch)
            best_ep = cfg.reset_epoch

        for epoch in range(best_ep + 1, cfg.num_epochs + 1):
            train(epoch)
            if cfg.val_interval > 0 and epoch % cfg.val_interval == 0:
                save_map = val_map(epoch)
                if save_map > best:
                    best = save_map
                    best_ep = epoch
                    print(saver.save(model.state_dict(), 'checkpoint'))
                else:
                    if epoch - best_ep > 30:
                        break
                print(saver.save(model.state_dict(), 'latestcheckpoint'))
        summary_writer.close()

if __name__ == '__main__':
    with DisablePrint(local_rank=cfg.local_rank):
        main()
