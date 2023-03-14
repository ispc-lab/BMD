## [BMD: Class-balanced Multicentric Dynamic Prototype Strategy for SFDA [ECCV 2022]](https://arxiv.org/abs/2204.02811).


#### Attention: Our new work on source-free universal domain adaptation has been accepted by CVPR-2023! The paper "Upcycling Models under Domain and Category Shift" is available at [https://arxiv.org/abs/2303.07110](https://arxiv.org/abs/2303.07110). The code also has been made public at [https://github.com/ispc-lab/GLC](https://github.com/ispc-lab/GLC).


The official repository of our paper "BMD: A General Class-balanced Multicentric Dynamic Prototype Strategy for Source-free Domain Adaptation". Here, we present the demo implementation on VisDA-C dataset.


### Prerequisites
- python3
- pytorch >= 1.7.0
- torchvision
- numpy, scipy, sklearn, PIL, argparse, tqdm, wandb

### Step
1. Please first prepare the pytorch enviroment.
2. Please download the [VisDA-C dataset](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) from the official website, and then unzip the dataset to the `./data` folder.
3. Prepare the source model by running following command
    > sh ./scripts/train_soruce.sh
4. Perform the target model adaptation by running following command. Please note that you need to first assign the source model checkpoint path in the `./scripts/train_target.sh` script.
    > sh ./scripts/train_target.sh

### Acknowledgement
This codebase is based on [SHOT-ICML2020](https://github.com/tim-learn/SHOT).

### Citation
If you find it helpful, please consider citing:
```
@inproceedings{sanqing2022BMD,
  title={BMD: A General Class-balanced Multicentric Dynamic Prototype Strategy for Source-free Domain Adaptation},
  author={Sanqing Qu, Guang Chen, Jing Zhang, Zhijun Li, Wei He, Dacheng Tao},
  booktitle={European conference on computer vision},
  year={2022}
}
```
