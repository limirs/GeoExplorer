# GeoExplorer
PyTorch implementation of _GeoExplorer:Active Geo-localization with Curiosity-Driven Exploration_ (ICCV 2025)

## Running

Set-up:
```bash
conda create --name geoexplorer
conda activate geoexplorer
conda install python==3.11
conda env create -f environment.yml
```


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

