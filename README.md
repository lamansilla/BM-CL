# BM-CL

---

## Installation

First, clone this repository. Then, create the virtual environment and install the required dependencies:

```bash
git clone https://github.com/lamansilla/BM-CL
cd BM-CL
conda env create -f environment.yml
conda activate bmcl
```

---

## Prepare data

In `scripts/prepare_data/` you will find the scripts required to prepare specific datasets:

- **Waterbirds**: `create_waterbirds.py`
- **CelebA**: `create_celeba.py`
- **CheXpert**: `create_chexpert.py`

These scripts generate the corresponding CSV metadata files in the `metadata/` folder. 

*Note:* you must download each dataset beforehand, as these scripts do not handle dataset downloading.

---

## Run experiments

In `experiments/` you will find the folder, scripts and configuration files to run different experiments:

- **SOTA comparison**: comparison with ERM and state-of-the-art bias mitigation methods.
- **Ablation study**: ablation study to analyze the impact of each model component of BM-CL.