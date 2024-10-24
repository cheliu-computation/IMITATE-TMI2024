# IMITATE-TMI2024

[IMITATE: Clinical Prior Guided Self-supervised Learning via Hierarchical Vision-Language Alignment](https://ieeexplore.ieee.org/abstract/document/10646593), TMI 2024.

###  Installation
To clone this repository:
```
git clone https://github.com/cheliu-computation/IMITATE-TMI2024.git
```
To install Python dependencies:
```
pip install -r requirements.txt
```
All experiments are implemented on A100 GPU.

### Pre-train Dataset downloading
Datasets we used are as follows:
- **MIMIC-CXR**: We downloaded the [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) dataset as the radiographs. Paired medical reports can be downloaded in [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/mimic-cxr-reports.zip).

### Preprocessing
- First we follow [MGCA](https://github.com/HKU-MedAI/MGCA) preprocessing to extract a master csv includes all CXR scans associated with report. You can find in [Preprocessing](https://github.com/HKU-MedAI/MGCA/blob/main/mgca/preprocess/mimic_cxr.py). 
- Then, run 'ext_data.py' to extract all scans and save as a npy file. It will accelerate the pre-training stage.

### Pre-training
We pre-trained IMITATE on MIMIC-CXR using this command:
```

cd /IMITATE-TMI2024/pretrain
torchrun --nnodes=1 --nproc_per_node=2 main.py
```

### Finetune on downstream tasks
We evlauate the performance of IMITATE on four downstream tasks: image classification, object detection, semantic segmentation and image-text retrieval tasks. 

For image-text retrieval task, we follow [GLoRIA-ICCV2021](https://github.com/marshuang80/gloria), please follow their offical code to extract data and implement Image-Text Retrieval tasks.

For image classification, semantic segmentation and object detection, we follow [MGCA-NeurIPS2022](https://github.com/HKU-MedAI/MGCA) offical configuration and code. The dataset can be found in MGCA repository.
