# v2e Experiment Data README
[![License: CC BY 4.0](https://licensebuttons.net/l/by/4.0/80x15.png)](https://creativecommons.org/licenses/by/4.0/)

## General information or Introduction section

If you are using this data, please cite:

+ Y. Hu, S-C. Liu, and T. Delbruck. v2e: From Video Frames to Realistic DVS Events. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), URL: https://arxiv.org/abs/2006.07722, 2021
+ Y. Hu, T. Delbruck, S-C. Liu, "Learning to Exploit Multiple Vision Modalities by Using Grafted Networks" in The 16th European Conference on Computer Vision (ECCV), Online, 2020.

### Contact

Yuhuang Hu  
yuhuang.hu@ini.uzh.ch  
Institute of Neuroinformatics, University of Zurich and ETH Zurich

## Access

The dataset is stored at Zenodo platform, click [here]() to access.

## Data and file(s) overview

There are four files

+ `CVPRW_V2E_N_CALTECH_101_DATA.tar.gz` contains both training and validation data for reproducing NCaltech-101 experiments in the paper.
+ `mvsec_day_data.tar.gz` contains training examples that are taken from the NGA paper (the second item in the citation section).
+ `v2e_day_200hz_all.h5` contains raw v2e events generated using MVSEC dataset's `outdoor_day_1` dataset.
+ `v2e_mvsec_train_data.tar.gz` contains training data for reproducing MVSEC car detection dataset. For validation, please check out [MVSEC-NIGHTL21](https://github.com/SensorsINI/MVSEC-NIGHTL21) dataset.
