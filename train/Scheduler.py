from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, ReduceLROnPlateau
from collections import Counter
from .utils import WarmupMultiStepLR, ManualStepLR


def make_lr_scheduler(cfg, optimizer):
    if cfg.train.warmup:
        scheduler = WarmupMultiStepLR(optimizer, cfg.train.milestones, cfg.train.gamma, cfg.train.warm_factor, cfg.train.warm_iter, 'linear')
    elif cfg.train.scheduler == 'manual':
        scheduler = ManualStepLR(optimizer, milestones=cfg.train.milestones, gammas=cfg.train.gammas)
    elif cfg.train.scheduler == 'exp':
        scheduler = ExponentialLR(optimizer, gamma=cfg.train.gamma)
    elif cfg.train.scheduler == 'rlop':
        scheduler = ReduceLROnPlateau(optimizer, factor=cfg.train.gamma, patience=cfg.train.patience,
                                      threshold=cfg.train.threshold, threshold_mode=cfg.train.threshold_mode,
                                      cooldown=cfg.train.cooldown)
    else:
        scheduler = MultiStepLR(optimizer, milestones=cfg.train.milestones, gamma=cfg.train.gamma)
    return scheduler


def set_lr_scheduler(cfg, scheduler):
    if cfg.train.warmup:
        scheduler.milestones = cfg.train.milestones
    else:
        scheduler.milestones = Counter(cfg.train.milestones)
    scheduler.gamma = cfg.train.gamma
