# MFTrans

**MFTrans**: Multi-Resolution Fusion Transformer for Robust Tumor Segmentation in Whole Slide Images  
(WACV 2025 Accepted 🎉)

---

## Requirements

- **Python**: 3.10.6  
- **Torch**: 2.0.0  
- **CUDA**: 12.1  
- **PyTorch Lightning**: 2.2.1  

---

## Docker Container
docker container is available 

"docker pull yangsunggyu/skyang_wsi:0727"

---
## Directory Structure
```
├── ckpt
│   └── MFNet-model-epoch=018-val_loss=0.372819.ckpt
├── data
│   ├── camelyon16
│   │   └── test
│   │       ├── masks
│   │       │   └── tumor_001
│   │       │       └── metastasis
│   │       │           ├── tumor_001_metastasis_3808x_3136y_224tilesize_1000tissueLevel_79foregroundLevel_mask.gif
│   │       │           ├── tumor_001_metastasis_3808x_3360y_224tilesize_1000tissueLevel_69foregroundLevel_mask.gif
│   │       ├── tiles
│   │       │   └── tumor_001
│   │       │       └── metastasis
│   │       │           ├── tumor_001_metastasis_3808x_3136y_224tilesize_1000tissueLevel_79foregroundLevel.jpg
│   │       │           ├── tumor_001_metastasis_3808x_3360y_224tilesize_1000tissueLevel_69foregroundLevel.jpg
│   │       └── tiles_0
│   │           └── tumor_001
│   │               ├── metastasis
│   │               │   ├── tumor_001_metastasis_3808x_3136y_224tilesize_1000tissueLevel_79foregroundLevel.jpg
│   │               │   ├── tumor_001_metastasis_3808x_3360y_224tilesize_1000tissueLevel_69foregroundLevel.jpg
│   │               └── non_metastasis
│   │                   
│   ├── cmc
│   └── paip2019
├── DataProcessing
│   ├── cmc-data.py
│   ├── cmc_visualization2.py
│   ├── model.py
│   └── __pycache__
├── model
│   ├── ConvNeXt
│   │   ├── convNet.py
│   │   ├── ConvNeXt2.py
│   │   ├── ConvNeXt.py
│   │   ├── __init__.py
│   ├── GT.py
│   ├── __init__.py
│   └── MFNet.py
├── data.py
├── model.py
├── test.py
├── train.py
├── tree.txt
├── utils.py
└── wsi_utils.py
```

## How to Run

  python train.py --accelerator gpu --devices 0 --lr 0.00001 --epochs 50 --archi MFNet --data camelyon
  
  python test.py --accelerator gpu --devices 0 --archi MFNet --data camelyon


## Citation 
```
@inproceedings{your_wacv2025_citation,
  title={MFTrans: Multi-Resolution Fusion Transformer for Robust Tumor Segmentation in Whole Slide Images},
  author={Your Name, Other Authors},
  booktitle={WACV},
  year={2025}
}
```
