# [✨AAAI 2025] DRAT: Unsupervised Degradation Representation Aware Transform for Real-World Blind Image Super-Resolution
> [Sen Chen](https://github.com/KKKc3231), Hongying Liu, Chaowei Fang, Fanhua Shang, Yuanyuan Liu, Liang Wan, Dongmei Jiang, Yaowei Wang

> School of Artiffcial Intelligence, Xidian University


# News:
- Our **DRAT** is accepted by AAAI2025✨!

# Overview of DRAT
 
# Code File Descriptions

| File                                      | Description                                                  |
| ----------------------------------------- | ------------------------------------------------------------ |
| basicsr/archs/drat_arch.py                | Implementation of DRAT.                                      |
| basicsr/add/encoder.py                    | Implementation of aux and blur encoder.                      |
| basicsr/losses/basic_loss.py              | Definitions of the loss functions.                           |
| degradation/scripts/generate_blur_data.py | Code for generating training and test data for Isotropic  and Anisotropic Gaussian kernels. |
| inference/inference_drat.py               | Code for inferring Super-Resolution (SR) results by using DRAT. |
| options/test/xxx/xxx.yml                  | YML configuration file for testing.                          |
| options/train/xxx/xxx.yml                 | YML configuration file for training.                         |
| basicsr/utils.py                          | Various utilities.                                           |

You can use the following commands to train and test:

```sh
# For training:
python train.py -opt=options/train/DRAT/setting1/x4/train_DRAT_x4.yml

# For testing:
python test.py -opt=options/test/DRAT/setting1/x4/test_urban100.yml
```
