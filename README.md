# UPENA: A Unified Single-Stage End-to-End Framework for Multi-Trait Pig Phenotyping


## Installation

### Set up the Python environment

```bash
conda create -n UPENA python=3.7
conda activate UPENA

pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## Dataset

The dataset resources used in UPENA are organized under `PigImageData`.

The image data and corresponding annotations can be downloaded from the LISAP dataset released with CIEN-LWEN:

- [BaiduNetdisk](https://pan.baidu.com/share/init?surl=pTzheIyYFX-LRDYEtta4kw&pwd=hzau)
- [CIEN-LWEN](https://github.com/TXiang-lab/CIEN-LWEN)

For backfat thickness prediction, place `bf_data.json` under `PigImageData`.

Organize the data as follows:

```text
UPENA/
├── PigImageData/
│   ├── Train/
│   ├── Val/
│   ├── Test/
│   ├── Train_annotations.json
│   ├── Val_annotations.json
│   ├── Test_annotations.json
│   ├── weight_data.json
│   └── bf_data.json
├── Config/
├── dataset/
├── evaluator/
├── network/
└── ...
```

## Pretrained Models

The pretrained models for liveweight and backfat thickness prediction can be downloaded from the following links:
[Liveweight](https://github.com/TXiang-lab/UPENA/releases/download/v1.0.0/liveweight.pth) and
[Backfat](https://github.com/TXiang-lab/UPENA/releases/download/v1.0.0/backfat.pth)

```text
data/model/
├── liveweight/
│   └── liveweight.pth
└── backfat/
    └── backfat.pth
```

## Testing

### Liveweight

```bash
cd $ROOT
CUDA_VISIBLE_DEVICES=0 python test.py --config_file liveweight
```

### Backfat thickness

```bash
cd $ROOT
CUDA_VISIBLE_DEVICES=0 python test.py --config_file backfat
```

## Training

### Liveweight

```bash
cd $ROOT
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node 4 \
train_net_ddp.py \
--config_file liveweight \
--bs 32 \
--gpus 4
```

### Backfat thickness

```bash
cd $ROOT
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node 4 \
train_net_ddp.py \
--config_file backfat \
--bs 32 \
--gpus 4
```

## Acknowledgement

The LISAP dataset is available from the [CIEN-LWEN](https://github.com/TXiang-lab/CIEN-LWEN) project. We thank the authors of the open-source projects used in this implementation for their valuable contributions.

## License

This project is released under the GNU General Public License v3.0.
