gpus = [0, 1, 2, 3]
ct_score = 0.05

resume = True
save_ep = 5
eval_ep = 5

skip_eval = False

demo_path = ''

parallel = True
input_mode = 'img'


class train(object):
    dataset = 'train'
    batch_size = 80
    epoch = 300
    gamma = 0.5
    lr = 0.0001
    milestones = [80, 120, 150, 170]
    num_workers = 64
    optim = 'sgd'
    scheduler = ''
    warmup = False
    warm_iter = 5
    warm_factor = 0.3
    weight_decay = 0.0005
    weight_dict = {'ct_hm': 1, 't': 1, 'semantic': 1}


class val(object):
    dataset = 'val'
    batch_size = 1
    epoch = -1


class test(object):
    dataset = 'test'
    batch_size = 1
    epoch = -1
