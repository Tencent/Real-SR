import os
import mlflow
import math
import argparse
import random
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from data.data_sampler import DistIterSampler


import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    mlflow.log_params({'Batch Size': opt['datasets']['train']['batch_size'],
                       'Color Model': opt['datasets']['train']['color'],
                       'Pixel loss': opt['train']['pixel_criterion'],
                       'Pixel Weight Channel 0': opt['train']['pixel_weight_ch0'],
                       'Pixel Weight Channel 1': opt['train']['pixel_weight_ch1'],
                       'Pixel Weight Channel 2': opt['train']['pixel_weight_ch2'],
                       'Iterations': opt['train']['niter'],})
    mlflow.log_artifact('./options/df2k/train_bicubic_noise.yml')
    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benckmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    train_loader = None
    val_loader = None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            # print('\n\n\n\n\n\n\n\n', dataset_opt)
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))

    pix_ch0_epoch_losses = []
    pix_ch1_epoch_losses = []
    pix_ch2_epoch_losses = []
    pix_losses = []
    fea_epoch_losses = []
    gan_epoch_losses = []
    real_epoch_losses = []
    fake_epoch_losses = []

    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        pix_ch0_epoch_loss = 0
        pix_ch1_epoch_loss = 0
        pix_ch2_epoch_loss = 0
        fea_epoch_loss = 0
        gan_epoch_loss = 0
        real_epoch_loss = 0
        fake_epoch_loss = 0
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            #### training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            logs = model.get_current_log()
            pix_ch0_epoch_loss += logs['l_g_pix_ch0']/len(train_data)
            pix_ch1_epoch_loss += logs['l_g_pix_ch1']/len(train_data)
            pix_ch2_epoch_loss += logs['l_g_pix_ch2']/len(train_data)
            fea_epoch_loss += logs['l_g_fea']/len(train_data)
            gan_epoch_loss += logs['l_g_gan']/len(train_data)
            real_epoch_loss += logs['l_d_real']/len(train_data)
            fake_epoch_loss += logs['l_d_fake']/len(train_data)
            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            # validation
            if current_step % opt['train']['val_freq'] == 0 and rank <= 0 and val_loader is not None:
                avg_psnr = val_pix_err_f = val_pix_err_nf = val_mean_color_err = 0.0
                idx = 0
                for val_data in val_loader:
                    idx += 1
                    img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                    img_dir = os.path.join(opt['path']['val_images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(val_data)
                    model.test()

                    visuals = model.get_current_visuals()
                    sr_img = util.tensor2img(visuals['SR'])  # uint8
                    gt_img = util.tensor2img(visuals['GT'])  # uint8

                    # Save SR images for reference
                    save_img_path = os.path.join(img_dir,
                                                 '{:s}_{:d}.png'.format(img_name, current_step))
                    util.save_img(sr_img, save_img_path)

                    # calculate PSNR
                    crop_size = opt['scale']
                    gt_img = gt_img / 255.
                    sr_img = sr_img / 255.
                    cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    avg_psnr += util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)

                avg_psnr = avg_psnr / idx
                val_pix_err_f /= idx
                val_pix_err_nf /= idx
                val_mean_color_err /= idx

                # log
                logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                    epoch, current_step, avg_psnr))
                # tensorboard logger
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger.add_scalar('val_pix_err_f', val_pix_err_f, current_step)
                    tb_logger.add_scalar('val_pix_err_nf', val_pix_err_nf, current_step)
                    tb_logger.add_scalar('val_mean_color_err', val_mean_color_err, current_step)

            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

        if current_step < total_iters:
            pix_loss = pix_ch0_epoch_loss + pix_ch1_epoch_loss + pix_ch2_epoch_loss
            pix_loss /= len(train_loader)
            fea_epoch_loss /= len(train_loader)
            gan_epoch_loss /= len(train_loader)
            real_epoch_loss /= len(train_loader)
            fake_epoch_loss /= len(train_loader)
        
            mlflow.log_metrics({'Pixel Loss Channel 0': pix_ch0_epoch_loss,
                                'Pixel Loss Channel 1': pix_ch1_epoch_loss,
                                'Pixel Loss Channel 2': pix_ch2_epoch_loss,
                                'Pixel Loss': pix_loss,
                                'Feature Loss': fea_epoch_loss,
                                'GAN Loss': gan_epoch_loss,
                                'Fake Loss': fake_epoch_loss},
                               step=epoch)

    mlflow.log_param('Epochs', epoch)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')

def plot_losses(lst, fname):
    plt.figure()
    plt.scatter(list(range(len(lst))), lst)
    plt.plot(list(range(len(lst))), lst, linestyle='dashed', alpha=0.25)
    plt.savefig(fname=fname,
                dpi=300,
                bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    with mlflow.start_run(experiment_id=1):
        main()
        mlflow.log_artifacts('../experiments/Corrupted_noise')
