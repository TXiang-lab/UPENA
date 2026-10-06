from .base import *

model = 'CSNet'
trait = 'liveweight'
weight = 'liveweight.pth'

model_dir = 'data/model/liveweight'
record_dir = 'data/record/liveweight'
result_dir = 'data/result/liveweight'

train.dataset = 'train'
train.epoch = 350
train.batch_size = 32
train.lr = 0.001
train.gamma = 0.5
train.warmup = True
train.warm_iter = 10
train.warm_factor = 0.1
train.milestones = [80, 150, 210, 260, 300, 330, 340]

val.dataset = 'val'
test.dataset = 'test'
