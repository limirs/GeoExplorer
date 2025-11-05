# GeoExplorer
PyTorch implementation of _GeoExplorer: Active Geo-localization with Curiosity-Driven Exploration_ (ICCV 2025)

## Running

Set-up the environment:
```bash
conda create --name geoexplorer
conda activate geoexplorer
conda install python==3.11
conda env create -f environment.yml
```

Data preparation:
Please follow the repo of [GOMAA-Geo](https://github.com/mvrl/GOMAA-Geo) to do the data preprocessing for the Masa, xBD and MM-GAG datasets.
For the SwissView dataset, please download the images from the Huggingface Repo [SwissView](https://huggingface.co/datasets/EPFL-ECEO/SwissView).
To process the aerial images:
1) get patches
set `path="../data/swissview/swissview100_patches"`, `img_path="../data/swissview/SwissView100/"` for SwissView100, and set `path="../data/swissview/swissviewmonuments_patches"`, `img_path="../data/swissview/SwissViewMonuments/aerial_view"` for SwissViewMonuments.
```bash
python get_patches.py
```

2) get features for areial views
login with your Hugging Face token `login("HuggingFace_Token_Here")`; set `data_path="../data/swissview/swissview100_patches/patches/*"`, `save_path="../data/swissview/swissview100_sat_patches.npy"`
```bash
python get_sat_embeddings.py
```

3) get features for ground views (SwissViewMonuments only)
set `data_path="../data/swissview/SwissViewMonuments/ground_view/"`, `save_path="../data/swissview/swissviewmonuments_grd.npy"`
```bash
python get_grd_embeddings.py
```


Training and Validation:
To train the model for action-state modeling:
```bash
python pretrain.py
```

To train the model to do curiosity-driven exploration:
```bash
python train.py
```

To run inference, run the following command:
```bash
python validate.py
```


## Citation and Acknowledgements
We would like to thank the authors of GOMMA-Geo for providing the code basis of this work. If you find this work helpful, please consider citing:

```bibtex
@article{mi2025geoexplorer,
      title={{GeoExplorer}: Active Geo-localization with Curiosity-Driven Exploration}, 
      author={Li Mi and Manon Béchaz and Zeming Chen and Antoine Bosselut and Devis Tuia},
      year={2025},
      journal={arXiv preprint arXiv:2508.00152},
}
```
```bibtex
@article{sarkar2024gomaa,
  title={GOMAA-Geo: GOal Modality Agnostic Active Geo-localization},
  author={Sarkar, Anindya and Sastry, Srikumar and Pirinen, Aleksis and Zhang, Chongjie and Jacobs, Nathan and Vorobeychik, Yevgeniy},
  journal={arXiv preprint arXiv:2406.01917},
  year={2024}
}
```

