# :point_right: Super-Resolution visualization
## :honeybee: Urban100
[<img src="assert/urban_05.png" height="288px["/>](https://imgsli.com/Mjc0NjUz) [<img src="assert/urban_91.png" height="288px["/>](https://imgsli.com/Mjc0NjUy)
## :dolphin:M109
[<img src="assert/M109_1.png" height="380px["/>](https://imgsli.com/Mjc0NjU5) [<img src="assert/M109_2.png" height="380px["/>](https://imgsli.com/Mjc0NjYw) [<img src="assert/M109_3.png" height="380px["/>](https://imgsli.com/Mjc0NjYx)
 
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
