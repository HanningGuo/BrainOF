import os
import time
import utils
import torch
import datetime
import argparse
import functools
from models import transformer
import torch.distributed as dist
from calflops import calculate_flops
from torchvision.transforms import v2
from utils import create_logger, get_config
from engine_finetune import train_one_epoch, evaluate
from datasets import BrainSignalDataset, DataTransform
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.checkpoint import DefaultLoadPlanner
from models.transformer import ARNESSBlock, DintTransformerBlock
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, CPUOffload
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, get_model_state_dict, set_model_state_dict, StateDictOptions


def get_parse_option():
    parser = argparse.ArgumentParser('Brain-OF fine-tuning and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')

    # easy config modification
    parser.add_argument('--root', type=str, metavar='PATH', help='root path of dataset')
    parser.add_argument('--out_root', type=str, metavar='PATH', help='root path of output folder')
    parser.add_argument('--dataset', type=str, metavar='DATASET', help='train and evaluate the dataset')
    parser.add_argument('--batch_size', type=int, help="batch size for single GPU")
    parser.add_argument('--pretrained_model', type=str, metavar="CHECKPOINT", help='pretrained weight from checkpoint')
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
    train_dataset = BrainSignalDataset(root=cfg.root, datasets=cfg.dataset.name, split='train', transform=transform, modality_cfg=cfg.dataset.modality)
    val_dataset = BrainSignalDataset(root=cfg.root, datasets=cfg.dataset.name, split='val', transform=transform, modality_cfg=cfg.dataset.modality)
    test_dataset = BrainSignalDataset(root=cfg.root, datasets=cfg.dataset.name, split='test', transform=transform, modality_cfg=cfg.dataset.modality)

    if cfg.distribute.distributed_mode:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.DistributedSampler(val_dataset, shuffle=False)
        test_sampler = torch.utils.data.DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler, val_sampler, test_sampler = None, None, None

    train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=cfg.train.batch_size, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, multiprocessing_context="fork", drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, sampler=val_sampler, batch_size=cfg.train.batch_size, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, multiprocessing_context="fork", drop_last=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, sampler=test_sampler, batch_size=cfg.train.batch_size, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, multiprocessing_context="fork", drop_last=False)

    # ------ mixup and cutmix ------
    if cfg.aug.mixup_alpha > 0:
        cutmix = v2.CutMix(alpha=cfg.aug.cutmix_alpha)
        mixup = v2.MixUp(alpha=cfg.aug.mixup_alpha)
        mixup_fn = v2.RandomChoice([cutmix, mixup])
    else:
        mixup_fn = None

    # ------ create model ------
    model = getattr(transformer, cfg.model.name)(
        num_classes=cfg.model.num_classes,
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
        classifier_head_init_scale=cfg.model.init_scale,
    )
    logger.info(model)

    # ------ enable LoRA ------
    if cfg.model.use_lora:
        model.add_lora(add_lora_arness=True, add_lora_encoder=True, add_lora_attn=True, add_lora_ffn=False, add_lora_proj=True, lora_rank=cfg.model.lora_rank, lora_alpha=cfg.model.lora_alpha, lora_dropout=cfg.model.lora_dropout)
        for name, p in model.named_parameters():
            if "lora_" in name or name.startswith("head."):
                p.requires_grad = True
            else:
                p.requires_grad = False

    # ------ enable linear probing ------
    if cfg.model.enable_lp:
        if cfg.model.num_classes > 0:
            model.head = torch.nn.Sequential(torch.nn.Linear(model.embed_dim, 4 * model.embed_dim), torch.nn.GELU(), torch.nn.Linear(4 * model.embed_dim, cfg.model.num_classes))
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("head.")

    # ------ profile the computational performance ------
    # if dist.get_rank() == 0:
    #     flops, macs, n_params = calculate_flops(model=model, args=[train_dataset[0][0].unsqueeze(0), train_dataset[0][2].unsqueeze(0).bool()])
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
            auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={DintTransformerBlock, ARNESSBlock})
            if cfg.amp_enable:
                mp_policy = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
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
    linear_scaled_lr = cfg.train.optimizer.base_lr * actual_batch_size / 1024.0
    linear_scaled_warmup_lr = cfg.train.lr_scheduler.warmup_lr * actual_batch_size / 1024.0
    linear_scaled_min_lr = cfg.train.lr_scheduler.min_lr * actual_batch_size / 1024.0
    linear_scaled_aux_loss_free_alpha = cfg.train.aux_loss_free_alpha * actual_batch_size / 1024.0
    logger.info(f"base lr: {cfg.train.optimizer.base_lr}")
    logger.info(f"actual lr: {linear_scaled_lr}")
    logger.info(f"base aux loss free alpha: {cfg.train.aux_loss_free_alpha}")
    logger.info(f"actual aux loss free alpha: {linear_scaled_aux_loss_free_alpha}")
    logger.info(f"actual batch size: {actual_batch_size}")
    # no weight decay on bias, norm layers, layer scale, router gates, patch embedding and latents
    no_wd_keywords = ['gamma', 'norm', 'lambda_qk', 'patch_embed', 'router.gate', 'latents']
    if cfg.distribute.parallelism.upper() == 'DDP':
        # DDP
        assigner = utils.LayerDecayValueAssigner(list(cfg.train.optimizer.layer_decay ** i for i in reversed(range(model_without_ddp.num_layers + 2))), cfg.distribute.parallelism)
        parameters = utils.get_finetune_param_groups_ddp(model_without_ddp, linear_scaled_lr, cfg.train.optimizer.weight_decay, assigner.get_layer_id, assigner.get_scale, skip_keywords=no_wd_keywords)
    else:
        # FSDP
        assigner = utils.LayerDecayValueAssigner(list(cfg.train.optimizer.layer_decay ** i for i in reversed(range(model.num_layers + 2))), cfg.distribute.parallelism)
        parameters = utils.get_finetune_param_groups_fsdp(model, linear_scaled_lr, cfg.train.optimizer.weight_decay, assigner.get_layer_id, assigner.get_scale, skip_keywords=no_wd_keywords)
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
    if cfg.train.criterion.task == 'classification':
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.train.criterion.label_smoothing)
    elif cfg.train.criterion.task == 'prediction':
        criterion = torch.nn.MSELoss()
    else:
        raise NotImplementedError("The loss function is not defined! Please specify the 'classification' or 'prediction' in the yaml!")

    best_perf = float('inf') if cfg.train.main_metric in ['mae', 'rmse'] else -float('inf')

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
            if 'best_perf' in checkpoint:
                best_perf = checkpoint['best_perf']
        logger.info(f"=> loaded successfully '{cfg.model.resume}' (epoch {checkpoint['epoch'] + 1})")
        del checkpoint
        torch.cuda.empty_cache()

    # ------ use pretraining checkpoint ------
    if cfg.model.pretrained_model and (not cfg.model.resume):
        logger.info(f"==============> Finetune from checkpoint....................")
        if cfg.distribute.parallelism.upper() == 'DDP':
            # DDP
            checkpoint = torch.load(cfg.model.pretrained_model, map_location=device, weights_only=False)
            checkpoint_model = checkpoint['model']
            checkpoint_model = {k.replace('encoder.', ''): v for k, v in checkpoint_model.items() if k.startswith('encoder.')}
            msg = model_without_ddp.load_state_dict(checkpoint_model, strict=False)
        else:
            # FSDP
            checkpoint = dict()
            checkpoint['encoder'] = get_model_state_dict(model, options=StateDictOptions(full_state_dict=False, cpu_offload=cfg.distribute.offload_params))
            dist.checkpoint.load(dict(model=checkpoint), checkpoint_id=cfg.model.pretrained_model, planner=DefaultLoadPlanner(allow_partial_load=True))
            msg = set_model_state_dict(model, model_state_dict=checkpoint['encoder'], options=StateDictOptions(full_state_dict=False, cpu_offload=cfg.distribute.offload_params))
        logger.info(msg)
        del checkpoint
        torch.cuda.empty_cache()

    # ------ evaluate mode ------
    if cfg.eval_mode:
        torch.cuda.empty_cache()
        test_metrics_dict, test_loss = evaluate(model, criterion, test_dataloader, device, cfg, logger, cfg.train.metrics)
        logger.info('test ' + ''.join(f'{key}: {value:.4f}, ' for key, value in test_metrics_dict.items()) + f'loss: {test_loss:.4f}')
        logger.info(f"========> Done!!!")
        return

    # ------ start training ------
    logger.info("==============> Start training.............................")
    start_time = time.time()
    for epoch in range(cfg.train.start_epoch, cfg.train.epochs):
        if cfg.distribute.distributed_mode:
            train_dataloader.sampler.set_epoch(epoch)

        train_metrics_dict, train_loss = train_one_epoch(model, criterion, train_dataloader, optimizer, lr_scheduler, loss_scaler, linear_scaled_aux_loss_free_alpha, epoch, device, cfg, logger, cfg.train.metrics, mixup_fn)

        if cfg.distribute.parallelism.upper() == 'DDP':
            # ddp save checkpoint
            if dist.get_rank() == 0 and (epoch % cfg.save_freq == 0 or epoch == (cfg.train.epochs - 1)):
                utils.save_checkpoint_ddp(cfg, epoch, model_without_ddp, best_perf, optimizer, lr_scheduler, loss_scaler, 'latest')
        elif epoch % cfg.save_freq == 0 or epoch == (cfg.train.epochs - 1):
            # fsdp save checkpoint
            utils.save_checkpoint_fsdp(cfg, epoch, model, best_perf, optimizer, lr_scheduler, loss_scaler, 'latest')

        # evaluation
        val_metrics_dict, val_loss = evaluate(model, criterion, val_dataloader, device, cfg, logger, cfg.train.metrics)

        perf = val_metrics_dict[cfg.train.main_metric]
        is_better = best_perf >= perf if cfg.train.main_metric in ['mae', 'rmse'] else best_perf <= perf
        if is_better:
            best_perf = perf
            test_metrics_dict, test_loss = evaluate(model, criterion, test_dataloader, device, cfg, logger, cfg.train.metrics)
            if cfg.distribute.parallelism.upper() == 'DDP':
                # ddp
                if dist.get_rank() == 0:
                    utils.save_checkpoint_ddp(cfg, epoch, model_without_ddp, best_perf, optimizer, lr_scheduler, loss_scaler, 'best')
            else:
                # fsdp
                utils.save_checkpoint_fsdp(cfg, epoch, model, best_perf, optimizer, lr_scheduler, loss_scaler, 'best')

        # print and record the metrics
        logger.info("#" + f"-" * 80 + "#\n")
        logger.info(f'Epoch: {epoch + 1:03d}\n'
                    'train ' + ''.join(f'{key}: {value:.4f}, ' for key, value in train_metrics_dict.items()) + f'loss: {train_loss:.4f}\n'
                    'val ' + ''.join(f'{key}: {value:.4f}, ' for key, value in val_metrics_dict.items()) + f'loss: {val_loss:.4f}\n'
                    'test ' + ''.join(f'{key}: {value:.4f}, ' for key, value in test_metrics_dict.items()) + f'loss: {test_loss:.4f}\n'
                    )
        logger.info(f'Best performance in the val dataset: {best_perf:.4f}\n')
        logger.info("#" + f"-" * 80 + "#\n")
        if writer is not None:
            writer.add_scalars(
                cfg.train.main_metric,
                {f'train {cfg.train.main_metric}': train_metrics_dict[cfg.train.main_metric],
                 f'val {cfg.train.main_metric}': val_metrics_dict[cfg.train.main_metric]},
                epoch
            )
            writer.add_scalars(
                'loss',
                {'train loss': train_loss,
                 'val loss': val_loss},
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
    writer, logger = create_logger(cfg, dist_rank=dist.get_rank(), name=cfg.model.name)
    logger.info(cfg.dump())

    # ------ main ------
    main(cfg)
