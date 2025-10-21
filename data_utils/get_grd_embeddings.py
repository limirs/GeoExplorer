from PIL import Image
import torch
from tqdm import tqdm
import numpy as np
import glob
import fire
from transformers import CLIPVisionModelWithProjection, CLIPProcessor
import os

def get_ground_embeddings(data_path="../data/swissview/ground_images/",
                       save_path="../data/swissview/grd_swisstopo.npy",
                       device="cuda:0"):

    model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)
    grd_embeddings = {}
    image_list = sorted(os.listdir(data_path))
    print(image_list)

    for i in range(len(image_list)):
        img = Image.open(data_path+image_list[i])
        inputs = processor(images=img,return_tensors="pt", padding=True)
        inputs.to(device)
        outputs = model(**inputs)
        
        image_embeds = outputs.image_embeds.squeeze(0).detach().cpu().numpy()
        grd_embeddings[f"img_{i}"] = np.array(image_embeds)

    np.save(save_path, grd_embeddings)


if __name__ == '__main__':
    
    fire.Fire(get_ground_embeddings)