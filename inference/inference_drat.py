import argparse
import cv2
import glob
import numpy as np
import os
import torch
from torch import nn as nn
from torch.nn import functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import default_init_weights, make_layer, pixel_unshuffle
from basicsr.add.encoder import Encoder
from basicsr.archs.arch_util import Upsample
import torch.nn as nn
import torchvision.models
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from basicsr.archs.drat_arch import DRATNet

# load dual-encoder for degradation representation
encoder_aux = Encoder()
encoder_blur = Encoder()

encoder_aux.load_state_dict(torch.load("/home/xdu_temp/DRAT/checkpoints/Encoder/encoder_aux.pt")) # 
encoder_blur.load_state_dict(torch.load("/home/xdu_temp/DRAT/checkpoints/Encoder/encoder_x2_x4_div2krk.pt"))

encoder_aux.to('cuda')
encoder_blur.to('cuda')

encoder_aux.eval()
encoder_blur.eval()

# inference 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default= "/home/xdu_temp/DRAT/checkpoints/DRAT/Setting2/DRAT_x4_div2krk_paper.pth"  # noqa: E501
    )
    parser.add_argument('--input', type=str, default="/data/cs/code/LR/0.2/", help='input test image folder')
    parser.add_argument('--output', type=str, default="/data/cs/code/LR/result/", help='output folder')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = DRATNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=10, num_grow_ch=24)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=False)
    model.eval()
    model = model.to(device)

    os.makedirs(args.output, exist_ok=True)
    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing123', idx, imgname)
        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)
        # inference
        try:
            with torch.no_grad():
                output = model(img)
        except Exception as error:
            print('Error', error, imgname)
        else:
            # save image
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            cv2.imwrite(os.path.join(args.output, f'{imgname}.png'), output)


if __name__ == '__main__':
    main()
