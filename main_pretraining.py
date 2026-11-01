import os
import time
import utils
import torch
import datetime
import argparse
import functools
from models import mtfm, loss
import torch.distributed as dist
from calflops import calculate_flops
from utils import create_logger, get_config
from engine_pretraining import train_one_epoch
from timm.scheduler.cosine_lr import CosineLRScheduler
from datasets import BrainSignalDataset, DataTransform
from torch.distributed.device_mesh import init_device_mesh
from models.transformer import ARNESSBlock, DintTransformerBlock
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, StateDictOptions


def get_parse_option():
    parser = argparse.ArgumentParser('Brain-OF pretraining script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')

    # easy config modification
    parser.add_argument('--root', type=str, metavar='PATH', help='root path of dataset')
    parser.add_argument('--out_root', type=str, metavar='PATH', help='root path of output folder')
    parser.add_argument('--dataset', type=str, metavar='DATASET', help='pretrain the dataset')
    parser.add_argument('--batch_size', type=int, help="batch size for single GPU")
    parser.add_argument('--resume', type=str, metavar="CHECKPOINT", help='resume from checkpoint')
    parser.add_argument('--accumulation_steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--disable_amp', action='store_true', help='disable pytorch amp')
    parser.add_argument('--eval_mode', action='store_true', help='perform evaluation only')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')

    args, _ = parser.parse_known_args()

    cfg = get_config(args)

    return args, cfg


def main(cfg):
    # ------ fix the seed for reproducibility ------
    if cfg.seed is not None:
        seed = cfg.seed + dist.get_rank()
        utils.set_seed(seed)

    # ------ data augmentation ------
    transform = DataTransform(signal_size=cfg.dataset.signal_size, patch_size=cfg.model.patch_size, max_seq_length=cfg.model.max_seq_length)

    # ------ create dataset ------
    train_dataset = BrainSignalDataset(root=cfg.root, datasets=cfg.dataset.name, split='train', pretraining=True, transform=transform, modality_cfg=cfg.dataset.modality)

    if cfg.distribute.distributed_mode:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=cfg.train.batch_size, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, multiprocessing_context="fork", drop_last=True)

    # ------ create model ------
    model = getattr(mtfm, cfg.model.name)(
        mask_ratio_min=cfg.pretraining.mask_ratio_min,
        mask_ratio_max=cfg.pretraining.mask_ratio_max,
        mask_ratio_mu=cfg.pretraining.mask_ratio_mu,
        mask_ratio_std=cfg.pretraining.mask_ratio_std,
        drop_path_rate=cfg.model.droppath,
        drop_path_type=cfg.model.droppath_type,
        max_seq_length=cfg.model.max_seq_length,
        qk_scale=cfg.model.qk_scale,
        ffn_dropout_rate=cfg.model.dropout,
        attn_dropout_rate=cfg.model.attn_drop,
        layer_scale_type=cfg.model.layer_scale_type,
        layer_scale_init_value=cfg.model.init_values,
        use_checkpoint=cfg.model.use_checkpoint,
        checkpoint_every_n=cfg.model.checkpoint_every_n,
        checkpoint_blk_ids=cfg.model.checkpoint_blk_ids,
        weight_init_style=cfg.model.weight_init_style,
    )
    logger.info(model)

    # ------ profile the computational performance ------
    # if dist.get_rank() == 0:
    #     flops, macs, n_params = calculate_flops(model=model, args=[train_dataset[0][0].unsqueeze(0), train_dataset[0][1].unsqueeze(0).bool(), train_dataset[0][2].unsqueeze(0), tuple(torch.tensor(elem).unsqueeze(0) for elem in train_dataset[0][3])])
    #     logger.info(f'{cfg.model.name} (Model size: {n_params}\t FLOPs: {flops})')

    device = torch.device(cfg.device)
    model.to(device)

    # ------ distribute the model ------
    if cfg.distribute.distributed_mode:
        if cfg.distribute.parallelism.upper() == 'DDP':
            # DDP
            model_without_ddp = model
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True, device_ids=[cfg.distribute.local_rank])
        else:
            # FSDP
            device_mesh = init_device_mesh("cuda", (cfg.distribute.world_size,))
            offload_policy = CPUOffload(offload_params=cfg.distribute.offload_params)
            # auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1e7)
            auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={DintTransformerBlock, ARNESSBlock})
            if cfg.amp_enable:
                mp_policy = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
            else:
                mp_policy = None
            model = FSDP(model, use_orig_params=True, device_id=torch.cuda.current_device(), cpu_offload=offload_policy, auto_wrap_policy=auto_wrap_policy, mixed_precision=mp_policy, device_mesh=device_mesh)

    loss_scaler = utils.NativeScalerWithGradNormCount()

    # ------ build optimizer ------
    # linear scale the learning rate according to total batch size, may not be optimal
    if cfg.train.accumulation_steps > 1:
        actual_batch_size = cfg.train.batch_size * dist.get_world_size() * cfg.train.accumulation_steps
        n_iter_per_epoch = len(train_dataloader) // cfg.train.accumulation_steps
    else:
        actual_batch_size = cfg.train.batch_size * dist.get_world_size() if cfg.distribute.distributed_mode else cfg.train.batch_size
        n_iter_per_epoch = len(train_dataloader)
    linear_scaled_lr = cfg.train.optimizer.base_lr * actual_batch_size / 7168.0
    linear_scaled_warmup_lr = cfg.train.lr_scheduler.warmup_lr * actual_batch_size / 7168.0
    linear_scaled_min_lr = cfg.train.lr_scheduler.min_lr * actual_batch_size / 7168.0
    linear_scaled_aux_loss_free_alpha = cfg.train.aux_loss_free_alpha * actual_batch_size / 7168.0
    logger.info(f"base lr: {cfg.train.optimizer.base_lr}")
    logger.info(f"actual lr: {linear_scaled_lr}")
    logger.info(f"base aux loss free alpha: {cfg.train.aux_loss_free_alpha}")
    logger.info(f"actual aux loss free alpha: {linear_scaled_aux_loss_free_alpha}")
    logger.info(f"actual batch size: {actual_batch_size}")
    logger.info(f"train dataset size: {len(train_dataset)}")
    # no weight decay on bias, norm layers, layer scale, router gates, latent queries and the patch embedding
    no_wd_keywords = ['gamma', 'norm', 'lambda_qk', 'patch_embed', 'router.gate', 'latents']
    if cfg.distribute.parallelism.upper() == 'DDP':
        parameters = utils.set_weight_decay_ddp(model_without_ddp, cfg.train.optimizer.weight_decay, skip_keywords=no_wd_keywords)
    else:
        parameters = utils.set_weight_decay_fsdp(model, cfg.train.optimizer.weight_decay, skip_keywords=no_wd_keywords)
    optimizer = torch.optim.AdamW(parameters, lr=linear_scaled_lr, betas=cfg.train.optimizer.betas, eps=cfg.train.optimizer.eps)

    # ------ build scheduler ------
    num_steps = int(cfg.train.epochs * n_iter_per_epoch)
    warmup_steps = int(cfg.train.lr_scheduler.warmup_epochs * n_iter_per_epoch)
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=(num_steps - warmup_steps) if cfg.train.lr_scheduler.warmup_prefix else num_steps,
        cycle_mul=1.,
        lr_min=linear_scaled_min_lr,
        warmup_lr_init=linear_scaled_warmup_lr,
        warmup_t=warmup_steps,
        cycle_limit=1,
        warmup_prefix=cfg.train.lr_scheduler.warmup_prefix,
    )

    # ------ build loss function ------
    criterion = loss.TimeFrequencyLoss()

    # ------ resume training ------
    if cfg.model.resume and os.path.exists(cfg.model.resume):
        logger.info(f"==============> Resume from checkpoint....................")
        if cfg.distribute.parallelism.upper() == 'DDP':
            # DDP resume
            checkpoint = torch.load(cfg.model.resume, map_location=device, weights_only=False)
            msg = model_without_ddp.load_state_dict(checkpoint['model'])
        else:
            # FSDP resume
            checkpoint = dict()
            checkpoint['model'], checkpoint['optimizer'] = get_state_dict(model, optimizer, options=StateDictOptions(full_state_dict=False, cpu_offload=cfg.distribute.offload_params))
            dist.checkpoint.load(dict(model=checkpoint['model'], optimizer=checkpoint['optimizer']), checkpoint_id=cfg.model.resume)
            msg = set_state_dict(model, optimizer, model_state_dict=checkpoint['model'], optim_state_dict=checkpoint['optimizer'], options=StateDictOptions(full_state_dict=False, cpu_offload=cfg.distribute.offload_params))
            checkpoint = torch.load(cfg.model.resume + '.pth', map_location=device, weights_only=False)
        logger.info(msg)
        if 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            if cfg.distribute.parallelism.upper() == 'DDP':
                # DDP optimizer resume
                optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            cfg.defrost()
            cfg.train.start_epoch = checkpoint['epoch'] + 1
            cfg.freeze()
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f"=> loaded successfully '{cfg.model.resume}' (epoch {checkpoint['epoch'] + 1})")
        del checkpoint
        torch.cuda.empty_cache()

    # ------ start training ------
    logger.info("===================> Start training....................")
    start_time = time.time()
    for epoch in range(cfg.train.start_epoch, cfg.train.epochs):
        if cfg.distribute.distributed_mode:
            train_dataloader.sampler.set_epoch(epoch)

        train_loss = train_one_epoch(model, criterion, train_dataloader, optimizer, lr_scheduler, loss_scaler, linear_scaled_aux_loss_free_alpha, epoch, device, cfg, logger)

        if cfg.distribute.parallelism.upper() == 'DDP':
            # DDP save checkpoint
            if dist.get_rank() == 0 and (epoch % cfg.save_freq == 0 or epoch == (cfg.train.epochs - 1)):
                utils.save_checkpoint_pretraining_ddp(cfg, epoch, model_without_ddp, optimizer, lr_scheduler, loss_scaler, 'latest')
        elif epoch % cfg.save_freq == 0 or epoch == (cfg.train.epochs - 1):
            # FSDP save checkpoint
            utils.save_checkpoint_pretraining_fsdp(cfg, epoch, model, optimizer, lr_scheduler, loss_scaler, 'latest')

        # print and record the metrics
        logger.info("#" + f"-" * 80 + "#\n")
        logger.info(f'Epoch: {epoch + 1:03d}\n'
                    f'train loss: {train_loss:.4f}\n'
                   )
        logger.info("#" + f"-" * 80 + "#\n")
        if writer is not None:
            writer.add_scalars(
                'loss',
                {'train loss': train_loss},
                epoch
            )

    # ------ training end ------
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Training time: {total_time_str}')
    logger.info(f"========> Done!!!")
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    # ------ create args and cfg ------
    args, cfg = get_parse_option()

    # ------ initial distributed training ------
    utils.init_distributed_mode(cfg)

    # ------ create tensorboard writer and logger ------
    writer, logger = create_logger(cfg, dist_rank=dist.get_rank(), name=cfg.model.name, pretraining=True)
    logger.info(cfg.dump())

    # ------ main ------
    main(cfg)
