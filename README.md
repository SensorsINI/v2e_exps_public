# v2e Experiments

The code release for reproduce experiments in the paper "v2e: From Video Frames to Realistic DVS Events"

## Citation

If you use this repository, please cite:

+ Y. Hu, S-C. Liu, and T. Delbruck. v2e: From Video Frames to Realistic DVS Events. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), URL: https://arxiv.org/abs/2006.07722, 2021.

## Contacts

Please submit an issue or contact us if you need our support.

Yuhuang Hu  
yuhuang.hu@ini.uzh.ch

## Setup

1. Clone this project to your own machine:

    ```bash
    $ git clone https://github.com/SensorsINI/v2e_exps_public
    ```

2. Install locally

    ```bash
    $ cd v2e_exps_public
    $ python setup.py develop
    ```

3. Install dependencies

    ```bash
    $ pip install torch torchvision
    $ pip install pytorch-lightning
    $ pip install test_tube
    $ pip install h5py
    $ pip install numpy
    ```
    _The above list might not be complete. All extra dependencies could be easily installed from `pip`._

## Data

Please read the data description and download dataset from [here](./DATA_README.md). After download, the data should be decompressed to `$HOME/data` folder.

## N-Caltech Experiments

The [Makefile](./scripts/ncaltech_exps/Makefile) defines the experiments.

## MVSEC Experiments

The [Makefile](./scripts/mvsec_exps/Makefile) defines the experiments.

Note that because our group is still improving the Network Grafting Algorithm (NGA) for other projects, only the training sketch is presented here.
Given NGA is easy to code, it should be easy to complete/rewrite the code by yourself.
