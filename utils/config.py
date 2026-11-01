from yacs.config import CfgNode as CN


_C = CN()


# ------ basic settings ------
# the root path of dataset
_C.root = '<root-path-to-dataset>'
# the output root path of tensorboard writer and logger
_C.out_root = '<output-root-to-save-results>'
# pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.pin_memory = True
# the number of threads/workers for data loading
_C.num_workers = 8
# frequency to logging info
_C.print_freq = 1
# frequency to save checkpoint
_C.save_freq = 1
# Whether fix the running seed to remove randomness
_C.seed = None
# Enable Pytorch automatic mixed precision (amp).
_C.amp_enable = True
# perform evaluation only
_C.eval_mode = False
# device to use
_C.device = 'cuda'


# ------ dataset settings ------
_C.dataset = CN()
# dataset name
_C.dataset.name = None
# the number of brain regions or channels and the length of brain signal (time points)
_C.dataset.signal_size = [246, 1984]   # (num_channels, time_series_length)

# data modality
_C.dataset.modality = CN()
# fMRI datasets
_C.dataset.modality.fMRI = ['ABIDE1', 'ABIDE2', 'ABRIM', 'ADHD200', 'ADNI', 'AOMIC_ID1000', 'AOMIC_PIOP1', 'AOMIC_PIOP2', 'BOLD5000', 'CamCAN_fMRI_Rest', 'CamCAN_fMRI_Task', 'CHCP_fMRI_Rest', 'CHCP_fMRI_Task', 'CORR', 'DLBS', 'HAD', 'HCP_YA_fMRI_Restin', 'HCP_YA_fMRI_Task', 'LEMON_fMRI', 'MOUS_fMRI_Rest', 'MOUS_fMRI_Task', 'NADR', 'Narratives', 'NKI_RS', 'Paingen', 'QTIM', 'REST_meta_MDD', 'SALD', 'SpaceTop', 'UCLA_CNP', 'WAND_fMRI_Rest', 'WAND_fMRI_Task']
# EEG datasets
_C.dataset.modality.EEG = ['CHB', 'FACED', 'LEMON_EEG', 'PhysioNet', 'PREDICT', 'REEG_BACA', 'SEED', 'SEED_DV', 'SEED_SD', 'SEED_VIG', 'SEED_VII', 'SEED_VLA_VRW', 'SEEDV', 'TDBRAIN', 'TUAB', 'TUEG', 'WBCIC']
# MEG datasets
_C.dataset.modality.MEG = ['CamCAN_MEG_Rest', 'CamCAN_MEG_Task', 'HCP_YA_MEG_Restin', 'HCP_YA_MEG_Task', 'MOUS_MEG_Rest', 'MOUS_MEG_Task', 'OMEGA', 'WAND_MEG_Rest', 'WAND_MEG_Task']


# ------ model settings ------
_C.model = CN()
# model name
_C.model.name = 'brainof_base_50M_A22M'
# pretrained weight from checkpoint, could be overwritten by command line argument
_C.model.pretrained_model = None
# checkpoint to resume
_C.model.resume = None
# patch size
_C.model.patch_size = 64
# the number of class
_C.model.num_classes = 2
# the qk scale in the self-attention
_C.model.qk_scale = None
# drop out rate
_C.model.dropout = 0.
# the initial value of layer scale
_C.model.init_values = 1e-3
# the layer scale type
_C.model.layer_scale_type = 'per_channel'
# the attention drop out rate
_C.model.attn_drop = 0.
# the drop path rate
_C.model.droppath = 0.
# the drop path type
_C.model.droppath_type = 'progressive'
# whether to use checkpointing to save memory (trade compute for memory)
_C.model.use_checkpoint = False
# use checkpoint frequency
_C.model.checkpoint_every_n = 1
# the layer of using checkpoint, None means using all layers (List[int])
_C.model.checkpoint_blk_ids = None
# the method to initial the model weights
_C.model.weight_init_style = 'pytorch'
# the initial scale of classifier head
_C.model.init_scale = 0.1
# support the max sequence length
_C.model.max_seq_length = 2048
# whether to use LoRA to finetuning the model
_C.model.use_lora = False
# lora rank
_C.model.lora_rank = 32
# lora scaling factor
_C.model.lora_alpha = 64
# lora drop out
_C.model.lora_dropout = 0.05
# enable the linear probing when finetuning
_C.model.enable_lp = False


# ------ train settings ------
_C.train = CN()
# batch size
_C.train.batch_size = 1024
# the number of epoch
_C.train.epochs = 100
# the start epoch from scratch or from resume
_C.train.start_epoch = 0
# clip gradient norm
_C.train.clip_grad = 5.0
# auxiliary loss free strategy for the load balancing of MoE
_C.train.aux_loss_free_alpha = 1e-3
# Gradient accumulation steps
_C.train.accumulation_steps = 1
# evaluation metrics
_C.train.metrics = ['bac', 'aucpr', 'auroc']
# main metric for model selection
_C.train.main_metric = 'bac'

# optimizer
_C.train.optimizer = CN()
# optimizer epsilon
_C.train.optimizer.eps = 1e-8
# optimizer betas
_C.train.optimizer.betas = (0.9, 0.99)
# base learning rate
_C.train.optimizer.base_lr = 1e-4
# weight decay
_C.train.optimizer.weight_decay = 1e-3
# layer decay
_C.train.optimizer.layer_decay = 1.0

# lr scheduler
_C.train.lr_scheduler = CN()
_C.train.lr_scheduler.min_lr = 1e-6
_C.train.lr_scheduler.warmup_epochs = 5
_C.train.lr_scheduler.warmup_lr = 1e-6
_C.train.lr_scheduler.warmup_prefix = True

# loss function
_C.train.criterion = CN()
# label smoothing
_C.train.criterion.label_smoothing = 0.0
# training task/type
_C.train.criterion.task = 'classification'


# ------ pretraining settings ------
_C.pretraining = CN()
# min mask ratio for mask ratio generator
_C.pretraining.mask_ratio_min = 0.3
# max mask ratio for mask ratio generator
_C.pretraining.mask_ratio_max = 0.8
# normal distribution center for mask ratio generator
_C.pretraining.mask_ratio_mu = 0.55
# standard deviation for mask ratio generator
_C.pretraining.mask_ratio_std = 0.25


# ------ distribution settings ------
_C.distribute = CN()
# distribute on itp
_C.distribute.dist_on_itp = False
# use distributed training
_C.distribute.distributed_mode = False
# distribute parallelism method
_C.distribute.parallelism = 'DDP'
# global rank
_C.distribute.rank = -1
# world size
_C.distribute.world_size = 1
# local rank
_C.distribute.local_rank = -1
# url used to set up distributed training
_C.distribute.dist_url = "env://"
# distributed backend
_C.distribute.dist_backend = "nccl"
# offload parameters to CPU for FSDP
_C.distribute.offload_params = False


# ------ augmentation settings ------
_C.aug = CN()
# mixup alpha, mixup enabled if > 0
_C.aug.mixup_alpha = 1.0
# cutmix alpha, cutmix enabled if > 0
_C.aug.cutmix_alpha = 1.0


def update_config(config, args):
    config.defrost()
    config.merge_from_file(args.cfg)

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    # merge from specific arguments
    if _check_args('root'):
        config.root = args.root
    if _check_args('out_root'):
        config.out_root = args.out_root
    if _check_args('dataset'):
        config.dataset.name = args.dataset
    if _check_args('batch_size'):
        config.train.batch_size = args.batch_size
    if _check_args('pretrained_model'):
        config.model.pretrained_model = args.pretrained_model
    if _check_args('resume'):
        config.model.resume = args.resume
    if _check_args('accumulation_steps'):
        config.train.accumulation_steps = args.accumulation_steps
    if _check_args('disable_amp'):
        config.amp_enable = False
    if _check_args('eval_mode'):
        config.eval_mode = True
    if _check_args('dist_on_itp'):
        config.distribute.dist_on_itp = True
    if _check_args('world_size'):
        config.distribute.world_size = args.world_size

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)
    return config
