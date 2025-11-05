from PIL import Image
# from ..rvsa import mae_vitae_base_patch16_dec512d8b
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import glob
import fire
from transformers import CLIPVisionModelWithProjection, CLIPProcessor
from einops import rearrange
from huggingface_hub import login
import os

# login with your Hugging Face token
login("HuggingFace_Token_Here")

class SatPatches(Dataset):
    def __init__(self, path, patch_size=5):
        self.path = path
        self.patch_size=patch_size
        self.transform_overhead = transforms.Compose([
            transforms.Resize(224,interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.3670, 0.3827, 0.3338), (0.2209, 0.1975, 0.1988))
        ])

    def __len__(self):
        return self.patch_size**2

    def __getitem__(self, idx):
        img = Image.open(f"{self.path}/patch_{idx}.jpg")
        transformed_img = self.transform_overhead(img)
        return transformed_img
    

def get_sat_embeddings(data_path="../data/swissview/swissview100_patches/patches/*",
                       patch_size=5,
                       save_path="../data/swissview/swissview100_sat_patches.npy",
                       device="cuda:0",
                       num_workers=4):

    model = CLIPVisionModelWithProjection.from_pretrained("MVRL/Sat2Cap")
    model = model.to(device)
    sat_embeddings = {}
    file_list = sorted(glob.glob(data_path))
    for i in range(len(file_list)):
        dataset = SatPatches(
            file_list[i], patch_size=patch_size)
        predloader = DataLoader(dataset, batch_size=1, num_workers=num_workers)
        preds = []
        for idx, batch in tqdm(enumerate(predloader)):
            preds.append(model(batch.to(device)).image_embeds.squeeze(0).detach().cpu().numpy())
        sat_embeddings[f"img_{i}"] = np.array(preds)
    np.save(save_path, sat_embeddings)


if __name__ == '__main__':

    fire.Fire(get_sat_embeddings)