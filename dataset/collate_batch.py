from torch.utils.data.dataloader import default_collate
import torch


def collate_batch(batch):
    data_input = {}
    inp = {'inp': default_collate([b['inp'] for b in batch])}
    meta = default_collate([b['meta'] for b in batch])
    data_input.update(inp)
    data_input.update({'meta': meta})

    # collate detection
    ct_hm = default_collate([b['ct_hm'] for b in batch])
    semantic_mask = default_collate([b['semantic_mask'] for b in batch])

    max_len = torch.max(meta['ct_num'])
    batch_size = len(batch)
    wh = torch.zeros([batch_size, max_len, 2], dtype=torch.float)
    ct_cls = torch.zeros([batch_size, max_len], dtype=torch.int64)
    ct_ind = torch.zeros([batch_size, max_len], dtype=torch.int64)
    ct_01 = torch.zeros([batch_size, max_len], dtype=torch.bool)
    ct_img_idx = torch.zeros([batch_size, max_len], dtype=torch.int64)
    ts = torch.zeros([batch_size, max_len], dtype=torch.float)
    ts_mask = torch.zeros([batch_size, max_len], dtype=torch.float)
    for i in range(batch_size):
        ct_01[i, :meta['ct_num'][i]] = 1
        ct_img_idx[i, :meta['ct_num'][i]] = i

    if max_len != 0:
        wh[ct_01] = torch.Tensor(sum([b['wh'] for b in batch], []))
        # reg[ct_01] = torch.Tensor(sum([b['reg'] for b in batch], []))
        ct_cls[ct_01] = torch.LongTensor(sum([b['ct_cls'] for b in batch], []))
        ct_ind[ct_01] = torch.LongTensor(sum([b['ct_ind'] for b in batch], []))
        ts[ct_01] = torch.Tensor(sum([b['gt_t'] for b in batch], []))
        ts_mask[ct_01] = torch.Tensor(sum([b['gt_mask_t'] for b in batch], []))
    detection = {'ct_hm': ct_hm, 'semantic_mask': semantic_mask, 'ct_cls': ct_cls, 'ct_ind': ct_ind, 'ct_01': ct_01,
                 'ct_img_idx': ct_img_idx, 'gt_t': ts, 'gt_mask_t': ts_mask}
    data_input.update(detection)

    return data_input
