import torch.nn as nn
import torch
from .utils import FocalLoss, sigmoid
import time
import datetime
import tqdm



def collect_training(poly, ct_01):
    batch_size = ct_01.size(0)
    poly = torch.cat([poly[i][ct_01[i]] for i in range(batch_size)], dim=0)
    return poly


class NetworkWrapper(nn.Module):
    def __init__(self, cfg, net):
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.ct_crit = FocalLoss()
        self.w_crit = torch.nn.SmoothL1Loss(reduction='none')
        self.sematic_crit=torch.nn.BCELoss()
        self.weight_dict = cfg.train.weight_dict

    def forward(self, batch):
        output = self.net(batch['inp'],batch)
        if not self.training:
            return output

        scalar_stats = {}
        loss = 0.
        ct_01 = batch['ct_01'].byte()
        gt_ts = collect_training(batch['gt_t'], ct_01)
        gt_mask_ts = collect_training(batch['gt_mask_t'], ct_01)

        ct_loss = self.ct_crit(sigmoid(output['ct_hm']), batch['ct_hm'])
        scalar_stats.update({'ct_loss': ct_loss})
        loss += ct_loss * self.weight_dict['ct_hm']

        sematic_loss=self.sematic_crit(sigmoid(output['semantic_mask']),batch['semantic_mask'])
        scalar_stats.update({'sematic_loss': sematic_loss})
        loss += sematic_loss * self.weight_dict['semantic']

        t_loss = self.w_crit(output['pred_ts'], gt_ts)
        t_loss = t_loss * gt_mask_ts
        t_loss = t_loss.sum() / gt_mask_ts.sum().clamp(min=1e-8)
        scalar_stats.update({'trait_loss': t_loss})
        loss += t_loss * self.weight_dict['t']

        return output, loss, scalar_stats,{}


class Trainer(object):
    def __init__(self, network):
        network = network.cuda()
        self.network = network

    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def train(self, epoch, data_loader, optimizer, recorder):
        max_iter = len(data_loader)
        self.network.train()
        end = time.time()
        train_loss_stats = {}
        for iteration, batch in enumerate(data_loader):
            if batch is None:
                continue
            data_time = time.time() - end
            iteration = iteration + 1
            recorder.step += 1

            # batch = self.to_cuda(batch)
            output, loss, loss_stats, image_stats = self.network(batch)
            # training stage: loss; optimizer; scheduler
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)
            optimizer.step()

            # data recording stage: loss_stats, time, image_stats
            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)
            for k, v in loss_stats.items():
                train_loss_stats.setdefault(k, 0)
                train_loss_stats[k] += v

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            if iteration % 20 == 0 or iteration == (max_iter - 1):
                # print training state
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                training_state = training_state.format(eta_string, str(recorder), lr, memory)
                print(training_state)

                # record loss_stats and image_dict
                recorder.update_image_stats(image_stats)
                recorder.record('train')

        return train_loss_stats['trait_loss'] / max_iter

    def val(self, epoch, data_loader, evaluator=None, recorder=None):
        self.network.eval()
        torch.cuda.empty_cache()
        data_size = len(data_loader)
        total_time = 0
        for batch in tqdm.tqdm(data_loader):

            if batch is None:
                continue

            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].cuda()

            with torch.no_grad():
                start = time.time()
                output = self.network(batch)
                total_time += time.time() - start

            if output is None:
                continue
            if evaluator is not None:
                evaluator.evaluate(output, batch)


        print(total_time / len(data_loader), '{} FPS'.format(len(data_loader) / total_time))


        if evaluator is not None:
            result, fig_stat = evaluator.summarize()

        if recorder:
            if evaluator is not None:
                recorder.record_fig('val', epoch, fig_stat)


def make_trainer(cfg, network):
    network = NetworkWrapper(cfg, network)

    return Trainer(network)
