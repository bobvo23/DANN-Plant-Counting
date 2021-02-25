import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as DataLoader
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')
    # device = cpu if not GPU
    device, _ = prepare_device(config['n_gpu'])

    # setup data_loader instances with MNIST
    data_loader_source = config.init_obj('data_loader_CVPPP', DataLoader)
    valid_data_loader_source = data_loader_source.split_validation()

    # setup data_loader instances with MNIST
    data_loader_target = config.init_obj('data_loader_KOMATSUNA', DataLoader)
    valid_data_loader_target = data_loader_target.split_validation()

    # build model architecture, then print to console
    model = config.init_obj('UNET_ADAPT_arch', module_arch)

    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)

    # get function handles of loss and metrics
    loss_fn_class = getattr(module_loss, config['density_loss'])
    loss_fn_domain = getattr(module_loss, config['bce_loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    #trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer_CVPPP', torch.optim, [
        {'params': counter_model.upsample.parameters(), 'lr': 1e-3},
        {'params': counter_model.downsample.parameters(), 'lr': 1e-3},
        {'params': counter_model.adapt.parameters(), 'lr': 1e-4},
    ])
    lr_scheduler = config.init_obj(
        'lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model=model,
                      loss_fn_class=loss_fn_class,
                      loss_fn_domain=loss_fn_domain,
                      metric_ftns=metrics,
                      optimizer=optimizer,
                      config=config,
                      device=device,
                      data_loader_source=data_loader_source,
                      valid_data_loader_source=valid_data_loader_source,
                      data_loader_target=data_loader_target,
                      valid_data_loader_target=valid_data_loader_target,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':

    args = argparse.ArgumentParser(description='PyTorch Template')
    # These arguments are not provided in json file
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'],
                   type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int,
                   target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
