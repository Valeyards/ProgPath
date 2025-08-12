#  Pan-cancer outcome prediction via a unified weakly supervised deep learning model


## Pre-requisites

All experiments are run on a machine with
- 1 NVIDIA RTX A6000 GPU
- Python (Python 3.10) and Pyotrch (torch\==2.0.1)

## Installation
1. Install [Anaconda](https://www.anaconda.com/distribution/)

2. Clone this reposity and cd into the directory:
```shell
git clone https://github.com/Valeyards/ProgPath.git
cd ProgPath
```

3. Create a new environment and install dependencies:
```shell
conda create -n progpath python=3.10 -y --no-default-packages
conda activate progpath
pip install --upgrade pip
pip install -r requirements.txt
```

## Model Download
The ProgPath model can be accessed from [here](https://drive.google.com/file/d/1Qqgl1HwP8W2_unsBNRvcF9XAxjIy6KVf/view?usp=sharing)

## Image Processing Pipeline

### Extract Tiles from Whole Slide Images
Preprocess the slides following [CLAM](https://github.com/mahmoodlab/CLAM), including foreground tissue segmentation and stitching. 

### Extract Image Feature Embeddings
1. Download the pretrained [Virchow2 model weights](https://huggingface.co/paige-ai/Virchow2), put it to *./weights/* and load the model
```python
import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from PIL import Image

# need to specify MLP layer and activation function for proper init
model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
model = model.eval()

transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
```

2. Use Virchow2 to extract image embeddings
```python
image = Image.open("/path/to/your/image.png")
image = transforms(image).unsqueeze(0)  # size: 1 x 3 x 224 x 224

output = model(image)  # size: 1 x 261 x 1280

class_token = output[:, 0]    # size: 1 x 1280
patch_tokens = output[:, 5:]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those

# concatenate class token and average pool of patch tokens
embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560
```

## Basic Usage: Predict Patient Risk with ProgPath

Please refer to `demo.ipynb` for a demonstration. 

1. Load the ProgPath model
```python
from utils.utils import read_yaml
from utils.model_factory import load_model
import torch

cfg = read_yaml('projects/configs/cfg_progpath.yaml')
model = load_model(cfg)
model.load_state_dict(torch.load('weights/progpath.pt'), strict=True)

```

2. Load image and clinical data
```python
import pandas as pd
import random
import torch
from datasets.SurvivalBagDataset import process_clinical

def encode_cancer_type(cancer_type):
    cancer_types = ['blca', 'brca', 'cesc', 'crc', 'gbm', 'hnsc', 'rcc', 'lgg', 'lihc', 'luad', 'lusc', 'paad', 'skcm', 'stad', 'ucec']
    encoding = [0] * len(cancer_types)
    if cancer_type in cancer_types:
        encoding[cancer_types.index(cancer_type)] = 1
    return encoding

random.seed(1)
patient_csv = pd.read_csv('csvs/sr_crc.csv')

patient_data = random.sample(list(patient_csv['patient_id']), 1)
patient_data = patient_csv[patient_csv['patient_id'] == patient_data[0]]
patch_features = torch.load(f'features/{patient_data["filename"].values[0]}')
clinical_feature = process_clinical(patient_data, columns=['age', 'sex', 'stage'])['processed_df'].drop(columns=['filename']).values
clinical_feature = torch.from_numpy(clinical_feature).float()
cancer_type = 'crc'
cancer_encoding = torch.tensor(encode_cancer_type(cancer_type)).float()
```
3. Predict patient risk
```python
model.eval()
model.to('cuda')
patch_features = patch_features.to('cuda')
res = model(h=patch_features, clinical_feature=clinical_feature.to('cuda'), cancer_encoding=cancer_encoding.to('cuda'))
risk = res['bag_logits'][0][1:]
print('patient id:', patient_data['patient_id'].values[0], 'risk:', risk.item())
```

## Evaluation 

To reproduce the results in our paper, we provide a reproducible result on [SR-CRC](https://www.ebi.ac.uk/biostudies/studies/S-BIAD1285) dataset.
Please refer to `demo.ipynb` for a demonstration. 
* First download our processed SR-CRC frozen features [here](https://pan.baidu.com/s/17_CJyuy5C6eDNozGmVIHGw?pwd=2ymk)
* Put the extracted features to *./features/* 
* Run the following command:
```shell
python3 eval.py --config_path projects/configs/cfg_progpath.yaml
```
The C-index and log-rank p-value will be printed to the screen. 
```python
sr_crc cindex_now: 0.7765380443204711 pvalue: 2.496161383249256e-10
```

The computed risk scores for this cohort and the corresponding Kaplan-Meier curve will be stored at `exp_progpath/evaluation/sr_crc/`

## Acknowledgements
The project was built on many amazing repositories: Virchow, [CLAM](https://github.com/mahmoodlab/CLAM), and [PORPOISE](https://github.com/mahmoodlab/PORPOISE). We thank the authors and developers for their contributions.

## Issue
Please open new threads or address questions to yuanw@stu.scu.edu.cn or xiyue.wang.scu@gmail.com

## License

ProgPath is made available under the CC BY-NC-SA 4.0 License and is available for non-commercial academic purposes.

