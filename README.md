# UDBRNet: A Novel Uncertainty Driven Boundary Refined Network for Organ at Risk Segmentation


This repository will contain the source code related to the research paper titled "UDBRNet: A Novel Uncertainty Driven Boundary Refined Network for Organ at Risk Segmentation". 

### Paper
Paper: [Link](https://doi.org/10.1371/journal.pone.0304771)

### Contact
For any kind of help or collaboration, Email [Riad Hassan](https://riadhassan.github.io/).

### Necessary Packages and Versions
```
numpy~=1.26.2
matplotlib~=3.8.2
tqdm~=4.66.1
monai~=1.3.0
torch~=2.1.2
scipy~=1.13.0
```

### Run the code
```
python train.py --dataset 'dataset_name' --data_path 'dataset_directory' --model_name 'UDBRNet'
```

### Output
#### 3D view of segmented organs for SegThor dataset
![3D output for SegThor dataset](/Output/3D%20output%20SegThor.png "3D output from SegThor dataset")

#### 3D view of segmented organs for LCTSC dataset
![3D output for LCTSC dataset](/Output/3D%20output%20LCTSC.png "3D output from SegThor dataset")

#### Predicted (Green) and ground truth (Red) contoured output for SegThor dataset
![Contoured output for SegThor dataset](/Output/SegThor_contoured.png "3D output from SegThor dataset")

#### Predicted (Green) and ground truth (Red) contoured output for LCTSC dataset
![Contoured output for LCTSC dataset](/Output/LCTSC_contoured.png "3D output from SegThor dataset")

### Cite this work
```
@article{10.1371/journal.pone.0304771,
    doi = {10.1371/journal.pone.0304771},
    author = {Hassan, Riad AND Mondal, M. Rubaiyat Hossain AND Ahamed, Sheikh Iqbal},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {UDBRNet: A novel uncertainty driven boundary refined network for organ at risk segmentation},
    year = {2024},
    month = {06},
    volume = {19},
    url = {https://doi.org/10.1371/journal.pone.0304771},
    pages = {1-18},
    number = {6},
}
```
