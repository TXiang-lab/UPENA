from .base import *

model = 'CSNet'
trait = 'backfat'
weight = 'backfat.pth'

model_dir = 'data/model/backfat'
record_dir = 'data/record/backfat'
result_dir = 'data/result/backfat'

train.dataset = 'bf_train'
train.epoch = 350
train.batch_size = 32
train.lr = 0.001
train.gamma = 0.5
train.warmup = True
train.warm_iter = 10
train.warm_factor = 0.1
train.milestones = [80, 150, 210, 260, 300, 330, 340]

val.dataset = 'bf_val'
test.dataset = 'bf_test'

eval_ep = 350
