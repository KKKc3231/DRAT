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

- For more details about the dataset, please see [DatasetPreparation](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md), which has a detailed introduction.

- For Anisotropic Gaussian kernels degradation, you can download DIV2KRK test datasets from [DIV2KRK](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

- For real-world degradation test, you can download the [RealSR_V3(Nikon and Canon)](https://drive.google.com/file/d/17ZMjo-zwFouxnm_aFM6CUHBwgRrLZqIM/view), [DRealSR](https://drive.google.com/drive/folders/1_EjDsD2-bBXb0iRonbWI6ihuHquxdna1) and [ImageNet256_x4_sr](https://drive.google.com/file/d/1NhmpON2dB2LjManfX6uIj8Pj_Jx6N-6l/view) (from ResShift)
```python
cd degradation/scripts/
python extract_subimages.py # crop the images to subimages

# getting degradation training data
python generate_blur_traindata.py # for iso/aniso blind sr (for training)
python generate_realesrgan_data.py # for real-world sr (for training)
python generate_blur_testdata.py # for generate iso/aniso test datasets
```
## Train
- The alorithm is in `archs/drat_arch.py`.
- Please Modify `options/train/SettingX/xx.yml` to set path, iterations, and other parameters.
- To train the DRAT, run the commands below.
```python
python train.py -opt=options/train/DRAT/setting1/x4/train_DRAT_x4.yml
python train.py -opt=options/train/DRAT/setting2/x4/train_DRAT_div2krkx2.yml
python train.py -opt=options/train/DRAT/setting3/x4/train_DRATGAN.yml
```

# Test & Infer
- Before test, please modify the model and data path.
- To test the DRAT, run the commands below.
```python
python test.py -opt=options/test/DRAT/setting1/x4/test_urban100.yml
python test.py -opt=options/test/DRAT/setting2/x4/test_div2krkx4.yml
python test.py -opt=options/test/DRAT/setting2/x4/test_DRATGAN.yml
```
- To infer DRAT on other image folders.
- To infer the DRAT, run the commands below.
- cd `inference/`
```python
python inference_drat.py --input your_input_path --output your_save_path
```
## Other Super-Resolution Visualization
[<img src="assert/urban_05.png"/>](https://imgsli.com/Mjc0NjUz) [<img src="assert/urban_91.png"/?](https://imgsli.com/Mjc0NjUy)
[<img src="assert/M109_1.png" height="400px["/>](https://imgsli.com/Mjc0NjU5) [<img src="assert/M109_2.png" height="400px["/>](https://imgsli.com/Mjc0NjYw) [<img src="assert/M109_3.png" height="400px["/>](https://imgsli.com/Mjc0NjYx)
