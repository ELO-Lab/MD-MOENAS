# Improving Transferability of Multi-Objective Evolutionary Neural Architecture Search by Utilizing Multiple Datasets in Network Evaluations

[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE.md)

Ngoc Hoang Luong, Tu Do

In NICS'21.

## Installation

- Clone this repo:

```bash
git clone https://github.com/MinhTuDo/MD-MOENAS.git
cd MD-MOENAS
```

- Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 0. Prepare the NAS Benchmarks

- Follow the instructions [here](https://github.com/D-X-Y/NATS-Bench/blob/main/README.md) to install benchmark files for NATS-Bench.
- **Remember to properly set the benchmark paths in config files, default data path is ~/.torch.**

### 1. Search

#### [Topology Search Space](https://github.com/D-X-Y/AutoDL-Projects/blob/master/docs/NAS-Bench-201.md)

```shell
# Baseline MOENAS evaluating only on CIFAR-10 dataset
python search.py --console_log -sw --use_archive --search_space tss -dts cifar10 --efficiency flops --eval_dts ImageNet16-120

# MD-MOENAS evaluating on CIFAR-10 & CIFAR-100 datasets
python search.py--console_log -sw --use_archive --search_space tss -dts cifar10 -dts cifar100 --efficiency flops --eval_dts ImageNet16-120
```

#### [Size Search Space](https://github.com/D-X-Y/NATS-Bench/blob/main/README.md)

```shell
# Baseline MOENAS evaluating only on CIFAR-10 dataset
python search.py --console_log -sw --use_archive --search_space sss -dts cifar10 --efficiency params --eval_dts ImageNet16-120

# MD-MOENAS evaluating on CIFAR-10 & CIFAR-100 datasets
python search.py--console_log -sw --use_archive --search_space sss -dts cifar10 -dts cifar100 --efficiency params --eval_dts ImageNet16-120
```

To evaluate IGD score on pre-computed optimal front during the search, simply provide `--eval_igd` flag.

You change dataset for IGD evaluation by providing value for `--eval_dts`. Note that this will work only if `--eval_igd` flag is used.

To change efficiency objective, simply change `--efficiency` parameters. Available efficiency objectives are `params`, `flops` and `latency`

For customized search, additional configurations can be modified through yaml config files in `config` folder.

## Acknowledgement

Code inspired from:

- [NATS-Bench: Benchmarking NAS Algorithms for Architecture Topology and Size](https://github.com/D-X-Y/NATS-Bench),
- [Pymoo: Multi-Objective Optimization in Python](https://github.com/anyoptimization/pymoo),
- [Automated Deep Learning Projects (AutoDL-Projects)](https://github.com/D-X-Y/AutoDL-Projects)
