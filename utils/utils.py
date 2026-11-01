import os
import torch
import random
import numpy as np
from torch import inf
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import get_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def init_distributed_mode(cfg):
    cfg.defrost()
    if cfg.distribute.dist_on_itp:
        cfg.distribute.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        cfg.distribute.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        cfg.distribute.local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        cfg.distribute.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(cfg.distribute.local_rank)
        os.environ['RANK'] = str(cfg.distribute.rank)
        os.environ['WORLD_SIZE'] = str(cfg.distribute.world_size)
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        cfg.distribute.rank = int(os.environ["RANK"])
        cfg.distribute.world_size = int(os.environ['WORLD_SIZE'])
        cfg.distribute.local_rank = int(os.environ['LOCAL_RANK'])
        cfg.distribute.dist_url = "env://"
    elif 'SLURM_PROCID' in os.environ:
        cfg.distribute.rank = int(os.environ['SLURM_PROCID'])
        cfg.distribute.world_size = int(os.environ['SLURM_NTASKS'])
        cfg.distribute.local_rank = int(os.environ['SLURM_LOCALID'])
        cfg.distribute.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT']) 

        os.environ['LOCAL_RANK'] = str(cfg.distribute.local_rank)
        os.environ['RANK'] = str(cfg.distribute.rank)
        os.environ['WORLD_SIZE'] = str(cfg.distribute.world_size)
    else:
        print('Not using distributed mode')
        cfg.distribute.distributed_mode = False
        return

    cfg.distribute.distributed_mode = True
    cfg.freeze()

    print(f"Distributed init (RANK and WORLD_SIZE {cfg.distribute.rank}/{cfg.distribute.world_size}): {cfg.distribute.dist_url}, GPU {cfg.distribute.local_rank}")

    torch.cuda.set_device(cfg.distribute.local_rank)
    device = torch.device(f"cuda:{cfg.distribute.local_rank}")
    torch.distributed.init_process_group(backend=cfg.distribute.dist_backend, init_method=cfg.distribute.dist_url,
                                         world_size=cfg.distribute.world_size, rank=cfg.distribute.rank, device_id=device)
    torch.distributed.barrier()


def set_weight_decay_ddp(model, weight_decay=1e-5, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list or check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)  # no weight decay on bias, norm and layer scale
        else:
            has_decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': has_decay, 'weight_decay': weight_decay}]


def set_weight_decay_fsdp(model, weight_decay=1e-5, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if name.endswith(".bias") or name in skip_list or check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)  # no weight decay on bias, norm and layer scale
        else:
            has_decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': has_decay, 'weight_decay': weight_decay}]


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True, model=None):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            if isinstance(model, FSDP):
                # FSDP: grads are sharded, clip with the global norm across all ranks
                norm = model.clip_grad_norm_(clip_grad if clip_grad is not None else float('inf'))
            elif clip_grad is not None:
                assert parameters is not None
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_checkpoint_ddp(cfg, epoch, model_without_ddp, best_perf, optimizer, lr_scheduler, loss_scaler, ckpt_name):
    save_state = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'best_perf': best_perf,
        'epoch': epoch,
        'scaler': loss_scaler.state_dict(),
        'cfg': cfg,
    }

    if ckpt_name is None:
        ckpt_name = f'checkpoint-{epoch}.pth'
    else:
        ckpt_name = f'checkpoint-{ckpt_name}.pth'
    save_path = os.path.join(cfg.out_root, 'results', cfg.dataset.name, cfg.save_dir, ckpt_name)
    torch.save(save_state, save_path)


def save_checkpoint_pretraining_ddp(cfg, epoch, model_without_ddp, optimizer, lr_scheduler, loss_scaler, ckpt_name):
    save_state = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'scaler': loss_scaler.state_dict(),
        'cfg': cfg,
    }

    if ckpt_name is None:
        ckpt_name = f'checkpoint-{epoch}.pth'
    else:
        ckpt_name = f'checkpoint-{ckpt_name}.pth'
    save_path = os.path.join(cfg.out_root, 'results', 'Pretraining', cfg.save_dir, ckpt_name)
    torch.save(save_state, save_path)


def save_checkpoint_fsdp(cfg, epoch, model, best_perf, optimizer, lr_scheduler, loss_scaler, ckpt_name):
    if ckpt_name is None:
        ckpt_name = f'checkpoint-{epoch}'
    else:
        ckpt_name = f'checkpoint-{ckpt_name}'
    save_path = os.path.join(cfg.out_root, 'results', cfg.dataset.name, cfg.save_dir, ckpt_name)

    model_state, optim_state = get_state_dict(model, optimizer, options=dist.checkpoint.state_dict.StateDictOptions(full_state_dict=False, cpu_offload=cfg.distribute.offload_params))
    dist.checkpoint.save(dict(model=model_state, optimizer=optim_state), checkpoint_id=save_path)

    if dist.get_rank() == 0:
        save_state = {
            'lr_scheduler': lr_scheduler.state_dict(),
            'best_perf': best_perf,
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'cfg': cfg,
        }
        torch.save(save_state, save_path + '.pth')


def save_checkpoint_pretraining_fsdp(cfg, epoch, model, optimizer, lr_scheduler, loss_scaler, ckpt_name):
    if ckpt_name is None:
        ckpt_name = f'checkpoint-{epoch}'
    else:
        ckpt_name = f'checkpoint-{ckpt_name}'
    save_path = os.path.join(cfg.out_root, 'results', 'Pretraining', cfg.save_dir, ckpt_name)

    model_state, optim_state = get_state_dict(model, optimizer, options=dist.checkpoint.state_dict.StateDictOptions(full_state_dict=False, cpu_offload=cfg.distribute.offload_params))
    dist.checkpoint.save(dict(model=model_state, optimizer=optim_state), checkpoint_id=save_path)

    if dist.get_rank() == 0:
        save_state = {
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'cfg': cfg,
        }
        torch.save(save_state, save_path + '.pth')


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


@torch.no_grad()
def gather_tensor(tensor):
    gt = tensor.contiguous()
    tensors_gather = [torch.zeros_like(gt) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, gt)
    output = torch.cat(tensors_gather, dim=0)
    return output


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def get_num_layer(var_name, num_max_layer, parallelism):
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif "patch_embed" in var_name:
        return 0
    elif "arness" in var_name:
        return 0
    elif "DintTransformerEncoder" in var_name:
        if parallelism == 'DDP':
            # DDP
            layer_id = int(var_name.split('.')[1])
        else:
            # FSDP
            layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values, parallelism='DDP'):
        self.values = values
        self.parallelism = parallelism.upper()

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer(var_name, len(self.values), self.parallelism)


def get_finetune_param_groups_ddp(model, lr, weight_decay, get_layer_func, scales, skip_list=(), skip_keywords=()):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or check_keywords_in_name(name, skip_keywords):
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_layer_func is not None:
            layer_id = get_layer_func(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if scales is not None:
                scale = scales(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr,
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr,
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    return list(parameter_group_vars.values())


def get_finetune_param_groups_fsdp(model, lr, weight_decay, get_layer_func, scales, skip_list=(), skip_keywords=()):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or (name in skip_list) or check_keywords_in_name(name, skip_keywords):
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_layer_func is not None:
            layer_id = get_layer_func(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if scales is not None:
                scale = scales(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr,
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr,
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    return list(parameter_group_vars.values())


def bf16_safe_rfft(x, dim=-1, device='cuda'):
    with torch.amp.autocast(device, enabled=False):
        return torch.fft.rfft(x.float(), dim=dim, norm='ortho')


def bf16_safe_irfft(x, dim=-1, device='cuda'):
    with torch.amp.autocast(device, enabled=False):
        return torch.fft.irfft(x, dim=dim, norm='ortho')
