## [✨AAAI 2025] DRAT: Unsupervised Degradation Representation Aware Transform for Real-World Blind Image Super-Resolution
> [Sen Chen](https://github.com/KKKc3231), Hongying Liu, Chaowei Fang, Fanhua Shang, Yuanyuan Liu, Liang Wan, Dongmei Jiang, Yaowei Wang

> School of Artiffcial Intelligence, Xidian University
## News:
- Our **DRAT** is accepted by AAAI2025✨!
  
## Overview of DRAT
 <img src="assert/g299.png"/>

## Environment
This code is based on basicsr.

- python >= 3.9
- pytorch == 1.13.1
```sh
conda env create -n DRAT -f environment.yml

python setup.py develop
```

## Code File Descriptions

| File                                      | Description                                                  |
| ----------------------------------------- | ------------------------------------------------------------ |
| basicsr/archs/drat_arch.py                | Implementation of DRAT.                                      |
| basicsr/add/encoder.py                    | Implementation of aux and blur encoder.                      |
| basicsr/losses/basic_loss.py              | Definitions of the loss functions.                           |
| degradation/scripts/                      | Code for generating training and test data for Isotropic, Anisotropic Gaussian kernels and complex degradations.|
| inference/inference_drat.py               | Code for inferring Super-Resolution (SR) results by using DRAT. |
| options/test/xxx/xxx.yml                  | YML configuration file for testing.                          |
| options/train/xxx/xxx.yml                 | YML configuration file for training.                         |
| basicsr/utils.py                          | Various utilities.                                           |
| options/train/xxx/xxx.yml                 | YML configuration file for training.                         |
| moco/                                     | Using MoCo to train Encoder for degradation representation learning                           |
| checkpoints/                              | Pretrained models                          |

You can use the following commands to train and test:

## Dataset Prearation
- We use DIV2K and Flickr2K as our training datasets (totally 3450 images). 

- For more details about the dataset, please see [DatasetPreparation](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md), which has a detailed introduction
```python
cd degradation/scripts/
python extract_subimages.py # crop the images to subimages

# getting degradation training data
python generate_blur_traindata.py # for iso/aniso blind sr (for training)
python generate_realesrgan_data.py # for real-world sr (for training)
python generate_blur_testdata.py # for generate iso/aniso test datasets
```

- you can use the following code to generate the blind-sr dataset.
```python


```
```sh
# For training:
python train.py -opt=options/train/DRAT/setting1/x4/train_DRAT_x4.yml

# For testing:
python test.py -opt=options/test/DRAT/setting1/x4/test_urban100.yml

# For inference
cd inference
python inference_drat.py --input your_input_path --output your_save_path
```
