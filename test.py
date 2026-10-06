import argparse
import importlib
import os
import time

import torch
from tqdm import tqdm

from network.One_stage_prediction import OneStageNet
from train import make_recorder
from train.model_utils.utils import load_network
from evaluator import make_evaluator
from dataset.data_loader import make_test_loader


parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default='liveweight')
args = parser.parse_args()

cfg = importlib.import_module('Config.' + args.config_file)

network = OneStageNet()
test_loader = make_test_loader(cfg, False, dataset_name=cfg.test.dataset)
evaluator = make_evaluator(cfg, dataset_name=cfg.test.dataset)
recorder = make_recorder(cfg)

model_path = os.path.join(cfg.model_dir, cfg.weight)
if not os.path.isfile(model_path):
    raise FileNotFoundError(
        'Pretrained weight not found: {}. Please place {} in {}.'
        .format(model_path, cfg.weight, cfg.model_dir)
    )

loaded_epoch = load_network(network, model_dir=model_path)

network = network.cuda()
network.eval()
count = 0
total_time = 0.0

for batch in tqdm(test_loader):
    if batch is None:
        continue

    for k in batch:
        if k != 'meta':
            batch[k] = batch[k].cuda()

    with torch.no_grad():
        start = time.time()
        output = network(batch['inp'])
        total_time += time.time() - start

        if output is None:
            continue

        evaluator.evaluate(output, batch)
        count += 1

if count == 0:
    raise RuntimeError('No valid samples were evaluated.')

print(total_time / count, '{} FPS'.format(count / total_time))
result, fig_stat = evaluator.summarize()
recorder.record_fig('test', loaded_epoch, fig_stat)
