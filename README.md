## BMD: Class-balanced Multicentric Dynamic Prototype Strategy for SFDA.

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
@article{sanqing2022BMD,
  title={BMD: A General Class-balanced Multicentric Dynamic Prototype Strategy for Source-free Domain Adaptation},
  author={Sanqing Qu, Guang Chen, Jing Zhang, Zhijun Li, Wei He, Dacheng Tao},
  journal={arXiv preprint arXiv:2204.02811},
  year={2022}
}
```