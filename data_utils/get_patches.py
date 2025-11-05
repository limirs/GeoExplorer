from PIL import Image
import torch
import numpy as np
import os
import fire
import glob
import pandas as pd
import tifffile

def get_patches(img_path, patch_size=5, save_path="patches_images"):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((1500,1500),Image.BICUBIC)
    img_array = torch.from_numpy(np.array(img))

    if img_array.shape[0]%patch_size!=0:
        raise ValueError

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    h, w = img_array.shape[0], img_array.shape[1]
    patch = img_array.unfold(0, h//patch_size, h//patch_size).unfold(1, w//patch_size, w//patch_size).reshape(-1, 3, h//patch_size, w//patch_size).numpy()
    for i in range(patch_size**2):
        img_save = Image.fromarray(patch[i, :, :, :].transpose(1, 2, 0)).resize((300,300),Image.BICUBIC)
        img_save.save(os.path.join(save_path, f"patch_{i}.jpg"))

def get_patches_multiple_images(path="../data/swissview/swissview100_patches",
                                img_path="../data/swissview/SwissView100/",
                                patch_size=5):

    filelist = sorted(os.listdir(img_path))
    print(filelist)
    for i in range(len(filelist)):
        file = filelist[i]
        get_patches(os.path.join(img_path, file), save_path=os.path.join(path, f"patches/img_{i}"), patch_size=patch_size)

if __name__=='__main__':
    fire.Fire(get_patches_multiple_images)
