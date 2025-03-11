import os
import sys
import cv2
import numpy as np
import torch
import os
import time



import utils as util
from u import imresize


def generate_mod_LR_bic(dataset,scale):
    # set parameters
    up_scale = scale
    mod_scale = scale
    sigs=[[0.80,1.60],[1.35,2.40],[1.80,3.20]]
    # set data dir
    #print("path",os.path.pardir)
    #time.sleep(60)
    sourcedir =  os.path.abspath(os.path.join('/data/cs/BlindSR/','datasets',dataset))
    savedir =  os.path.abspath(os.path.join('/data/cs/BlindSR/','datasets',dataset+'x4_test'))
    
    # load PCA matrix of enough kernel
    print("load PCA matrix")
    pca_matrix = torch.load(
        os.path.abspath(os.path.join("/data/cs/BlindSR/checkpoints/Pca_matrix/pca_matrix.pth")), map_location=lambda storage, loc: storage
    )
    print("PCA matrix shape: {}".format(pca_matrix.shape))

    degradation_setting = {
        "random_kernel": False,
        "code_length": 10,
        "ksize": 21,
        "pca_matrix": pca_matrix,
        "scale": up_scale,
        "cuda": True,
        "rate_iso": 1.0,
        #"noise": True,
        #"noise_high": 0.0588 # level 15: 15 / 255. | level 30: 30 / 255.
    }

    # set random seed
    util.set_random_seed(0)

    saveHRpath = os.path.join(savedir, "HR", "x" + str(mod_scale))
    saveLRblurpath = os.path.join(savedir, "LRblur", "x" + str(up_scale))

    if not os.path.isdir(sourcedir):
        print("Error: No source data found")
        exit(0)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    if not os.path.isdir(os.path.join(savedir, "HR")):
        os.mkdir(os.path.join(savedir, "HR"))
    if not os.path.isdir(os.path.join(savedir, "LRblur")):
        os.mkdir(os.path.join(savedir, "LRblur"))

    if not os.path.isdir(saveHRpath):
        os.mkdir(saveHRpath)
    else:
        print("It will cover " + str(saveHRpath))

    if not os.path.isdir(saveLRblurpath):
        os.mkdir(saveLRblurpath)
    else:
        print("It will cover " + str(saveLRblurpath))

    filepaths = sorted([f for f in os.listdir(sourcedir) if f.endswith(".png")])
    print(filepaths)
    num_files = len(filepaths)

    # kernel_map_tensor = torch.zeros((num_files, 1, 10)) # each kernel map: 1*10

    # prepare data with augementation
    
    for i in range(num_files):
        filename = filepaths[i]
        print("No.{} -- Processing {}".format(i, filename))
        # read image
        image = cv2.imread(os.path.join(sourcedir, filename))

        width = int(np.floor(image.shape[1] / mod_scale))
        height = int(np.floor(image.shape[0] / mod_scale))
        # modcrop
        if len(image.shape) == 3:
            image_HR = image[0 : mod_scale * height, 0 : mod_scale * width, :]
        else:
            image_HR = image[0 : mod_scale * height, 0 : mod_scale * width]
        # LR_blur, by random gaussian kernel
        img_HR = util.img2tensor(image_HR)
        C, H, W = img_HR.size()
        
        for sig in np.linspace(0.2,3.8,13):
            prepro = util.SRMDPreprocessing(sig=sig, **degradation_setting)
            LR_img, ker_map = prepro(img_HR.view(1, C, H, W))
            image_LR_blur = util.tensor2img(LR_img)
            cv2.imwrite(os.path.join(saveLRblurpath, 'sig{}_{}'.format(sig,filename)), image_LR_blur)
            cv2.imwrite(os.path.join(saveHRpath, 'sig{}_{}'.format(sig,filename)), image_HR)

        # kernel_map_tensor[i] = ker_map
    # save dataset corresponding kernel maps
    # torch.save(kernel_map_tensor, './Set5_sig2.6_kermap.pth')
    print("Image Blurring & Down smaple Done: X" + str(up_scale))


if __name__ == "__main__":
    scales=[4]
    datasets=['Set5']
    for scale in scales:
        for dataset in datasets:
            generate_mod_LR_bic(dataset,scale)