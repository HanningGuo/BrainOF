import time
import utils
import torch
import datetime
from utils import metrics_fn
from timm.utils import AverageMeter
from models.transformer import Router
from typing import Callable, Iterable


def train_one_epoch(
        model: Callable,
        criterion,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        lr_scheduler,
        loss_scaler,
        aux_loss_free_alpha,
        epoch: int,
        device: torch.device,
        cfg,
        logger,
        metrics: list = ['bac', 'aucpr', 'auroc'],
        mixup_fn: Callable = None,
):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    metrics_meter = {key: AverageMeter() for key in metrics}

    start = time.time()
    end = time.time()
    for idx, (signals, targets, attn_masks) in enumerate(data_loader):
        signals = signals.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        attn_masks = attn_masks.to(device, non_blocking=True)

        if mixup_fn is not None:
            signals, targets = mixup_fn(signals.unsqueeze(1), targets)
            signals = signals.squeeze(1)

        # forward
        with torch.amp.autocast(cfg.device, dtype=torch.bfloat16, enabled=cfg.amp_enable):
            outputs = model(signals, attn_masks.bool())
            loss = criterion(outputs, targets)
        loss = loss / cfg.train.accumulation_steps

        grad_norm = loss_scaler(loss, optimizer, clip_grad=cfg.train.clip_grad, parameters=model.parameters(), update_grad=(idx + 1) % cfg.train.accumulation_steps == 0, model=model)
        if (idx + 1) % cfg.train.accumulation_steps == 0:
            optimizer.zero_grad()
            lr_scheduler.step((epoch * num_steps + idx) // cfg.train.accumulation_steps)

        # auxiliary loss free strategy for load balancing
        # update bias AFTER optimizer step
        for m in model.modules():
            if isinstance(m, Router):
                m.update_bias(aux_loss_free_alpha=aux_loss_free_alpha)

        metrics_dict = metrics_fn(outputs, targets, metrics=metrics, num_classes=cfg.model.num_classes)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        for key in metrics:
            metrics_meter[key].update(metrics_dict[key].item(), targets.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % cfg.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch + 1}/{cfg.train.epochs}][{idx + 1}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'{metrics[0]} {metrics_meter[metrics[0]].val:.4f} ({metrics_meter[metrics[0]].avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch + 1} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    return {key: metrics_meter[key].avg for key in metrics}, loss_meter.avg


@torch.no_grad()
def evaluate(
        model,
        criterion,
        data_loader,
        device: torch.device,
        cfg,
        logger,
        metrics: list = ['bac', 'aucpr', 'auroc'],
):
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    metrics_meter = {key: AverageMeter() for key in metrics}

    start = time.time()
    end = time.time()
    for idx, (signals, targets, attn_masks) in enumerate(data_loader):
        signals = signals.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        attn_masks = attn_masks.to(device, non_blocking=True)

        # compute output
        with torch.amp.autocast(cfg.device, dtype=torch.bfloat16, enabled=cfg.amp_enable):
            outputs = model(signals, attn_masks.bool())
            loss = criterion(outputs, targets)

        # measure performance and record loss
        loss = utils.reduce_tensor(loss)
        loss_meter.update(loss.item(), targets.size(0))

        metrics_dict = metrics_fn(outputs, targets, metrics=metrics, num_classes=cfg.model.num_classes)
        for key in metrics:
            metrics_dict[key] = utils.reduce_tensor(metrics_dict[key])
            metrics_meter[key].update(metrics_dict[key].item(), targets.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % cfg.print_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx + 1}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'{metrics[0].upper()} {metrics_meter[metrics[0]].val:.4f} ({metrics_meter[metrics[0]].avg:.4f})\t'
                f'Mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"Testing takes {datetime.timedelta(seconds=int(epoch_time))}")
    return {key: metrics_meter[key].avg for key in metrics}, loss_meter.avg
