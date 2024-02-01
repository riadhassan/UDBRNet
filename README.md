<span style="color:red"> *Currently the paper is under review. After publication, the source code and pre-trained model will be released.* </span>

# UDBRNet: A Novel Uncertainty Driven Boundary Refined Network for Organ at Risk Segmentation


This repository will contain the source code related to the research paper titled "UDBRNet: A Novel Uncertainty Driven Boundary Refined Network for Organ at Risk Segmentation". 

### Necessary Packages and Versions
```
torch~=1.13.1
torchvision~=0.14.1
monai~=1.3.0
nibabel~=5.1.0
vedo~=2023.5.0
numpy~=1.26.2
scipy~=1.11.4
Pillow~=10.1.0
scikit-image~=0.22.0
matplotlib~=3.8.2
tqdm~=4.66.1
pandas~=2.1.4
```

### Run the code (*Sourcecode will be available after publication of the paper*)
```
python main.py --dataset 'dataset_name' --data_path 'Dataset_directory' --arch 'UDBRNet'
```

### Output
#### 3D view of output from SegThor dataset
![3D output from SegThor dataset](/Output/3D%20output%20SegThor.png "3D output from SegThor dataset")

#### 3D view of output from LCTSC dataset
![3D output from LCTSC dataset](/Output/3D%20output%20LCTSC.png "3D output from SegThor dataset")

#### Contoured output from SegThor dataset
![3D output from SegThor dataset](/Output/2D_Sample_Output_SegThor.PNG "3D output from SegThor dataset")

#### Contoured output from LCTSC dataset
![3D output from SegThor dataset](/Output/2D_Sample_Output_LCTSC.PNG "3D output from SegThor dataset")