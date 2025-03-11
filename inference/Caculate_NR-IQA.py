# conda activate torch25
import pyiqa
import torch
from PIL import Image
import numpy as np
import argparse
import os
from collections import defaultdict

def read_image(img_path, ref_image=None):
    # Open image and convert to RGB
    img = Image.open(img_path).convert('RGB')
    
    # Resize the image to match the reference image dimensions if provided
    if ref_image is not None:
        w, h = img.size
        _, _, h_ref, w_ref = ref_image.shape  # Get height and width from reference image
        if w != w_ref or h != h_ref:
            img = img.resize((w_ref, h_ref), Image.ANTIALIAS)
    
    # Normalize pixel values to [0, 1]
    img = np.asarray(img) / 255.0
    img = torch.from_numpy(img).float()
    
    # Reorder dimensions to match PyTorch format (C, H, W)
    img = img.permute(2, 0, 1)
    
    # Move image to the specified device (GPU) and add batch dimension
    device = torch.device('cuda')
    img = img.to(device).unsqueeze(0)
    
    return img.contiguous()

# caculate niqe
def get_NIQE(enhanced_image, gt_path=None):
    niqe_metric = pyiqa.create_metric('niqe', device=enhanced_image.device).to(torch.device('cuda'))
    return  niqe_metric(enhanced_image)

# caculate brisque
def get_BRISQUE(enhanced_image, gt_path=None):
    brisque_metric = pyiqa.create_metric('brisque', device=enhanced_image.device).to(torch.device('cuda'))
    return  brisque_metric(enhanced_image)

# caculate musiq
def get_MUSIQ(enhanced_image, gt_path=None):
    MUSIQ_metric = pyiqa.create_metric('musiq', device=enhanced_image.device).to(torch.device('cuda'))
    return  MUSIQ_metric(enhanced_image)  

# caculate nrqm
def get_NRQM(enhanced_image, gt_path=None):
    NRQM_metric = pyiqa.create_metric('nrqm', device=enhanced_image.device).to(torch.device('cuda'))
    return  NRQM_metric(enhanced_image) 

# caculate ilniqe
def get_ILNIQE(enhanced_image, gt_path=None):
    ILNIQE_metric = pyiqa.create_metric('ilniqe', device=enhanced_image.device).to(torch.device('cuda'))
    return  ILNIQE_metric(enhanced_image) 

# caculate fid
def get_FID(enhanced_image_path, gt_path):
    fid_metric = pyiqa.create_metric('fid').to(torch.device('cuda'))
    score = fid_metric(enhanced_image_path, gt_path)
    
if __name__ == '__main__':
  realbasicvsr_dir =  "/data0/cs/DRAT_code/results/RealBasicVSR_inference/036/" #"/data0/cs/DRAT_code/results/RealBasicVSR_inference/036_n/"
  realviformer_dir = "/data0/cs/DRAT_code/results/Duibi/029_75000_realbasicvsrpp/" #vi:7.29 ai:7.7695 8.0434 7.766 #/data0/cs/DRAT_code/results/RealViformer_VideoLQ/013/" #"/data0/cs/DRAT_code/results/RealViformer_VideoLQ/036_n/"
  realavsr_dir = "/data0/cs/DRAT_code/results/RealAVSR/036/" #"/data0/cs/DRAT_code/results/RealAVSR/036_n/"
  realdvsr_dir = "/data0/cs/DRAT_code/results/RealDVSR/036/"
  dratgan = "/data0/cs/DRAT_code/results/DRATGAN_90000/visualization/RealSR_Canon/" #3.16 3.02 3.09  NIQE iter_10000: 2.8892 iter_5000: 2.9951 iter_60000: 2.8266 ILNIQE: iter_10000: 18.4645  iter_5000: 18.7636
  sr_list =[realviformer_dir+os.listdir(realviformer_dir)[i] for i in range(len(os.listdir(realviformer_dir)))]
  sr_list1 =[dratgan+os.listdir(dratgan)[i] for i in range(len(os.listdir(dratgan)))]
  scores = 0
  for path in sr_list:  # 029: 10.9246 10.6479 10.6853
    img = read_image(path) # 036 dra 12.4722  dvsr 13.4452  realvi 13.50  12600_realviformer: [98.4201 100.9829  101.4336] 97.5474 108.2944： 75000:realdravsr 102.8586]
    scores += get_MUSIQ(img)
    print("ok")
  niqe_avg = scores / len(sr_list1)
  print("MUSIQ_avg", niqe_avg)
#   realbasicvsr = "/data0/cs/DRAT_code/results/RealBasicVSR_inference/036_n/00000001.png"
#   realviformer = "/data0/cs/DRAT_code/results/RealViformer_VideoLQ/036_n/00000001.png"
#   realavsr = "/data0/cs/DRAT_code/results/RealAVSR/036_n/00000001.png"
#   realbasicvsr_img = read_image(realbasicvsr)
#   realviformer_img = read_image(realviformer)
#   realavsr_img = read_image(realavsr)
#   realbasicvsr_niqe_score = get_MUSIQ(realbasicvsr_img)
#   realviformer_niqe_score = get_MUSIQ(realviformer_img)
#   realavsr_niqe_score = get_MUSIQ(realavsr_img)
#   print("realbasicvsr_niqe_score",realbasicvsr_niqe_score)
#   print("realviformer_niqe_score",realviformer_niqe_score)
#   print("realavsr_niqe_score",realavsr_niqe_score)