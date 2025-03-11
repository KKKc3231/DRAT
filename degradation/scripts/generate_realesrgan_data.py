# to generate the realesrgan data to train a Real-DRAT
# learn the degradation-mode from RealESRGAN
import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from torch.nn import functional as F
from PIL import Image
import torchvision.transforms as transforms
#import matplotlib.pyplot as plt
from torch.utils import data as data
from degradation import circular_lowpass_kernel, random_mixed_kernels 
from degradation import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from img_process_util import filter2D
from diffjpeg import DiffJPEG
# from multiprocessing import Pool
# from tqdm import tqdm
"""
    blur_kernel_size: 21
    kernel_list: ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
    kernel_prob: [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    sinc_prob: 0.1
    blur_sigma: [0.2, 3]
    betag_range: [0.5, 4]
    betap_range: [1, 2]

    blur_kernel_size2: 21
    kernel_list2: ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
    kernel_prob2: [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    sinc_prob2: 0.1
    blur_sigma2: [0.2, 1.5]
    betag_range2: [0.5, 4]
    betap_range2: [1, 2]

    final_sinc_prob: 0.8
"""

# config for func: BlurImgs()
config = {
    "scale": 4,
    # the first degradation process 
    "resize_prob": [0.2, 0.7, 0.1],  # up, down, keep
    "resize_range": [0.5, 1.5],
    "gaussian_noise_prob": 0.5,
    "noise_range": [1, 15],
    "poisson_scale_range": [0.05, 0.5],
    "gray_noise_prob": 0.4,
    "jpeg_range": [65, 95],
    # the second degradation process 
    "second_blur_prob": 0.2,
    "resize_prob2": [0.3, 0.4, 0.3],  # up, down, keep
    "resize_range2": [0.8, 1.2],
    "gaussian_noise_prob2": 0.5,
    "noise_range2": [1, 10],
    "poisson_scale_range2": [0.05, 0.2],
    "gray_noise_prob2": 0.4,
    "jpeg_range2": [75, 100]
}

# option for func: generate_random_kernels()
option = {
    "blur_kernel_size": 13,
    "kernel_list": ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
    "kernel_prob": [0.60, 0.40, 0.0, 0.0, 0.0, 0.0],
    "sinc_prob": 0.1,
    "blur_sigma": [0.2, 0.8],
    "betag_range": [1.0, 1.5],
    "betap_range": [1, 1.2],
    
    "blur_kernel_size2": 7,
    "kernel_list2": ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
    "kernel_prob2": [0.60, 0.40, 0.0, 0.0, 0.0, 0.0],
    "sinc_prob2": 0.0,
    "blur_sigma2": [0.2, 0.5],
    "betag_range2": [0.5, 0.8],
    "betap_range2": [1, 1.2],

    "final_sinc_prob": 0.2
}


device = 'cuda'
kernel_range = [2 * v + 1 for v in range(3, 11)]
pulse_tensor = torch.zeros(21,21).float()
pulse_tensor[10, 10] = 1
jpeger = DiffJPEG(differentiable=False).cuda()

def generate_random_kernels(opt=None):
    opt = option
    # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
    kernel_size = random.choice(kernel_range)

    if np.random.uniform() < opt['sinc_prob']: # 小于0.1则采用sinc核
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to = False)
    else:
        kernel = random_mixed_kernels(
            opt['kernel_list'],
            opt['kernel_prob'],
            kernel_size,
            opt['blur_sigma'],
            opt['blur_sigma'], [-math.pi, math.pi],
            opt['betag_range'],
            opt['betap_range'],
            noise_range = None
        )

    # pad kernel 
    pad_size = (21 - kernel_size) // 2 
    kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
    # print(kernel.shape)
    # # 可视化kernel
    # plt.imshow(kernel, cmap='gray')
    # plt.title('Kernel Visualization')
    # # plt.colorbar()
    # plt.show()
    # plt.savefig('/data/cs/BlindSR/degradation/kernel_visualization.png')

    # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
    kernel_size = random.choice(kernel_range)
    if np.random.uniform() < opt['sinc_prob2']: # 小于0.1则采用sinc核
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to = False)
    else:
        kernel2 = random_mixed_kernels(
            opt['kernel_list2'],
            opt['kernel_prob2'],
            kernel_size,
            opt['blur_sigma2'],
            opt['blur_sigma2'], [-math.pi, math.pi],
            opt['betag_range2'],
            opt['betap_range2'],
            noise_range = None
        )

    pad_size = (21 - kernel_size) // 2 
    kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))
    # print(kernel2.shape)
    # # 可视化kernel
    # plt.imshow(kernel2, cmap='gray')
    # plt.title('Kernel2 Visualization')
    # # plt.colorbar()
    # plt.show()
    # plt.savefig('/data/cs/BlindSR/degradation/kernel2_visualization.png')

    # ------------------------------------- the final sinc kernel ------------------------------------- #
    if np.random.uniform() < opt['final_sinc_prob']:
        kernel_size = random.choice(kernel_range)
        omega_c = np.random.uniform(np.pi / 3, np.pi)
        sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to = 21)
        sinc_kernel = torch.FloatTensor(sinc_kernel)
    else:
        sinc_kernel = pulse_tensor

    kernel = torch.FloatTensor(kernel)
    kernel2 = torch.FloatTensor(kernel2)
    return kernel, kernel2, sinc_kernel

# kernel, kernel2, sinc_kernel = generate_random_kernels(opt=option)

def BlurImgs(kernel1=None, kernel2=None, sinc_kernel=None, config=None, img_path=None):
    kernel = kernel1
    kernel2 = kernel2
    sinc_kernel = sinc_kernel
    config = config
    
    # use kernel to downsample image
    kernel = kernel.to(device)
    kernel2 = kernel2.to(device)
    sinc_kernel = sinc_kernel.to(device)

    image_path = img_path
    image = Image.open(image_path)

    # totensor
    transform = transforms.ToTensor()
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0) # 扩展维度
    image_tensor = image_tensor.to(device)
    ori_h, ori_w = image_tensor.size()[2:4]

    # ----------------------- The first degradation process ----------------------- #
    # first blur 
    out = filter2D(image_tensor, kernel)

    # random resize
    updown_type = random.choices(['up', 'down', 'keep'], config['resize_prob'])[0]
    if updown_type == 'up':
        scale = np.random.uniform(1, config['resize_range'][1])
    elif updown_type == 'down':
        scale = np.random.uniform(config['resize_range'][0], 1)
    else: # keep
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(out, scale_factor=scale,mode=mode)

    # add noise
    gray_noise_prob = config['gray_noise_prob']
    if np.random.uniform() < config['gaussian_noise_prob']:
        out = random_add_gaussian_noise_pt(
            out, sigma_range=config['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob
        )
    else:
        out = random_add_poisson_noise_pt(
            out, scale_range=config['poisson_scale_range'], gray_prob=gray_noise_prob, clip=True, rounds=False
        )

    # JPEG compression
    jpeg_p = out.new_zeros(out.size(0)).uniform_(*config['jpeg_range'])
    # print(updown_type, scale, jpeg_p)
    out = torch.clamp(out, 0, 1)
    out = jpeger(out, quality=jpeg_p)

    # ----------------------- The second degradation process ----------------------- #
    # second blur 
    if np.random.uniform() < config['second_blur_prob']:
        out = filter2D(out, kernel2)

    # random resize 
    updown_type = random.choices(['up', 'down', 'keep'], config['resize_prob2'])[0]
    if updown_type == 'up':
        scale = np.random.uniform(1, config['resize_range2'][1])
    elif updown_type == 'down':
        scale = np.random.uniform(config['resize_range2'][0], 1)
    else:
        scale = 1

    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(
        out, size=(int(ori_h / config['scale'] * scale), int(ori_w / config['scale'] * scale)), mode=mode
    )

    # add noise
    gray_noise_prob = config['gray_noise_prob2']
    if np.random.uniform() < config['gaussian_noise_prob2']:
        out = random_add_gaussian_noise_pt(
            out, 
            sigma_range=config['noise_range2'], 
            clip=True,
            rounds=False, 
            gray_prob=gray_noise_prob
        )
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=config['poisson_scale_range2'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False
        )

    # JPEG compression + the final sinc filter
    # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
    # as one operation.
    # We consider two orders:
    #   1. [resize back + sinc filter] + JPEG compression
    #   2. JPEG compression + [resize back + sinc filter]
    # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines
    # save the blur image 
    if np.random.uniform() < 0.5:
        # resize back + the final sinc filter
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(ori_h // config['scale'], ori_w // config['scale']), mode=mode)
        out = filter2D(out, sinc_kernel)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*config['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
    else:
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*config['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
        # resize back + the final sinc filter
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(ori_h // config['scale'], ori_w // config['scale']), mode=mode)
        out = filter2D(out, sinc_kernel)

    out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    return out 

if __name__ == '__main__':
    # souce_data = "/data/cs/BlindSR/datasets/BSD100"
    # save_data = "/data/cs/BlindSR/datasets/blur_bsd100"
    souce_data = "/data0/cs/datasets/train/DIF2K_sub/"
    save_data = "/data0/cs/datasets/train/DIF2K_sub_realesrgan_x4/"
    for idx, path in enumerate(os.listdir(souce_data)):
        img_path = os.path.join(souce_data,path)
        kernel, kernel2, sinc_kernel = generate_random_kernels(opt=option)
        out = BlurImgs(kernel, kernel2, sinc_kernel, config, img_path)
        To_pil = transforms.ToPILImage()
        out_blur = To_pil(out.squeeze(0).cpu()) # remove batchsize and to cpu
        out_blur.save(os.path.join(save_data,path))
        print("===== Processing the idx {} {}.=====".format(idx+1, path))


