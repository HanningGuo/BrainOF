import os
import sys
import logging
import datetime
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter


def create_logger(cfg, dist_rank=0, pretraining=False, name=''):
    # create the file name for tensorboard writer
    if pretraining:
        dataset = 'Pretraining'
    else:
        dataset = cfg.dataset.name
    save_dir = f'{cfg.model.name}_lr{cfg.train.optimizer.base_lr}_weightdecay{cfg.train.optimizer.weight_decay}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'

    cfg.defrost()
    cfg.save_dir = save_dir
    cfg.freeze()

    # create directory if no exist
    os.makedirs(os.path.join(cfg.out_root, 'results', dataset, save_dir), exist_ok=True)
    if dist_rank == 0:
        # create tensorboard writer for master process
        writer_path = os.path.join(cfg.out_root, 'results', dataset, save_dir)
        writer = SummaryWriter(writer_path)
    else:
        writer = None

    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s - %(name)s] - (%(filename)s - %(lineno)d): %(levelname)s - %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create file handle and set level
    logger_path = os.path.join(cfg.out_root, 'results', dataset, save_dir, f'log_rank{dist_rank}.log')
    file_handler = logging.FileHandler(logger_path)
    file_handler.setLevel(logging.DEBUG)
    # add formatter to file handle
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    # add file handle to logger
    logger.addHandler(file_handler)

    if dist_rank == 0:
        # create console handlers for master process
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    return writer, logger
