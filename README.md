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
@inproceedings{mi2025geoexplorer,
  title={GeoExplorer: Active Geo-localization with Curiosity-Driven Exploration},
  author={Mi, Li and B{\'e}chaz, Manon and Chen, Zeming and Bosselut, Antoine and Tuia, Devis},
  booktitle={ICCV},
  pages={6122--6131},
  year={2025}
}
```
```bibtex
@inproceedings{sarkar2024gomaa,
  title={Gomaa-geo: Goal modality agnostic active geo-localization},
  author={Sarkar, Anindya and Sastry, Srikumar and Pirinen, Aleksis and Zhang, Chongjie and Jacobs, Nathan and Vorobeychik, Yevgeniy},
  booktitle={NeurIPS},
  pages={104934--104964},
  year={2024}
}
```

