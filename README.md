# Robustness of Multi-Modal Foundational Models

Research code for evaluating the robustness of multi-modal foundational models (MMFMs) against adversarial attacks. This repository contains implementations for testing vision-language models like OpenFlamingo against sparse and non-sparse adversarial perturbations, as well as fine-tuning CLIP models on adversarial examples and COCO counterfactuals.

**Code adapted from:** [RobustVLM](https://github.com/chs20/RobustVLM)

## Table of Contents

- [Robustness of Multi-Modal Foundational Models](#robustness-of-multi-modal-foundational-models)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Dataset Setup](#dataset-setup)
    - [VLM Evaluation Datasets](#vlm-evaluation-datasets)
      - [1. VizWiz Dataset](#1-vizwiz-dataset)
      - [2. OK-VQA Dataset](#2-ok-vqa-dataset)
      - [3. Flickr30k Dataset](#3-flickr30k-dataset)
      - [4. COCO Dataset (2014)](#4-coco-dataset-2014)
    - [CLIP Fine-tuning Datasets](#clip-fine-tuning-datasets)
      - [1. COCO Counterfactuals (COCO-CFs)](#1-coco-counterfactuals-coco-cfs)
      - [2. APGD Adversarial Images](#2-apgd-adversarial-images)
      - [3. COCO 2017 Validation Set](#3-coco-2017-validation-set)
      - [4. COCO Captions and Classification Datasets](#4-coco-captions-and-classification-datasets)
  - [Usage](#usage)
    - [Sparse vs Non-Sparse Attacks Evaluation](#sparse-vs-non-sparse-attacks-evaluation)
      - [Configuration Options](#configuration-options)
      - [Running the Scripts](#running-the-scripts)
    - [Fine-tuning CLIP Models](#fine-tuning-clip-models)
      - [Parameters](#parameters)
      - [Running the Scripts](#running-the-scripts-1)
    - [Zero-Shot Image Classification](#zero-shot-image-classification)
      - [Parameters](#parameters-1)
      - [Running the Scripts](#running-the-scripts-2)
    - [Image-Text Retrieval](#image-text-retrieval)
      - [Parameters](#parameters-2)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Prerequisites

- **Python version:** 3.11.x
- **Java:** JDK 1.8.0_202 (required for CIDEr score computation)
- **CUDA-compatible GPU** (for model training and inference)

## Installation

1. Clone the repository and navigate to the project directory:
   ```bash
   cd Robust_mmfm
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the OpenFlamingo 9B model from [HuggingFace](https://huggingface.co/openflamingo/OpenFlamingo-9B-vitl-mpt7b). After downloading, it should be located in `$HOME/.cache/huggingface/hub/` with the name `models--openflamingo--OpenFlamingo-9B-vitl-mpt7b`.

4. Install [JDK 1.8.0_202](https://tubcloud.tu-berlin.de/s/YdRcyp888N5qwkx) and add it to your PATH:
   ```bash
   # Add to ~/.bashrc or ~/.zshrc
   export PATH=$PATH:/path/to/jdk1.8.0_202/bin
   export LANG=en_US.UTF-8
   ```

## Dataset Setup

### VLM Evaluation Datasets

#### 1. VizWiz Dataset
- Download the [VizWiz VQA dataset](https://vizwiz.org/tasks-and-datasets/vqa/) (train and validation sets)
- Annotation files are included in the repository, but can be re-downloaded if corrupted
- Place images in:
  - `./open_flamingo_datasets/VizWiz/train`
  - `./open_flamingo_datasets/VizWiz/val`

#### 2. OK-VQA Dataset
- Download the [OK-VQA dataset](https://okvqa.allenai.org/download.html) (training and testing images)
- Annotation files are included in the repository
- Place all images in: `./open_flamingo_datasets/OKVQA`

#### 3. Flickr30k Dataset
- Download using instructions from [awsaf49/flickr-dataset](https://github.com/awsaf49/flickr-dataset)
- Annotation files (`karpathy_flickr30k.json`, `dataset_flickr30k_coco_style.json`) are included
- Alternative annotation download: [TU Berlin Cloud](https://nc.mlcloud.uni-tuebingen.de/index.php/s/mtRnQFaZJkR9zaX)
- Place images in: `./open_flamingo_datasets/Flickr30k/Images`

#### 4. COCO Dataset (2014)
- Download [COCO 2014](https://cocodataset.org/#download) train and validation sets
- Annotation files are included in the repository
- Alternative annotation downloads:
  - [karpathy_coco.json](https://nc.mlcloud.uni-tuebingen.de/index.php/s/mtRnQFaZJkR9zaX)
  - [captions_val2014.json](https://github.com/tylin/coco-caption/blob/master/annotations/captions_val2014.json)
- Place images in:
  - `./open_flamingo_datasets/COCO/train2014`
  - `./open_flamingo_datasets/COCO/val2014`
### CLIP Fine-tuning Datasets

#### 1. COCO Counterfactuals (COCO-CFs)
- Download `images.zip` from [HuggingFace COCO-Counterfactuals](https://huggingface.co/datasets/Intel/COCO-Counterfactuals/tree/main/data)
- Unzip and place images in:
  - `./open_flamingo_datasets/COCO_CF/images`
  - `./clip_train_datasets/MS_COCO_COCO_CF/images`
- Copy original images (ending with `_0.jpg`) to:
  ```bash
  cp ./open_flamingo_datasets/COCO_CF/images/*_0.jpg ./clip_train_datasets/MS_COCO_APGD_4/images
  cp ./open_flamingo_datasets/COCO_CF/images/*_0.jpg ./clip_train_datasets/MS_COCO_APGD_1/images
  ```

#### 2. APGD Adversarial Images
- Download from [TU Berlin Cloud](https://tubcloud.tu-berlin.de/s/YdRcyp888N5qwkx):
  - `apgd_1_images.zip` → `./clip_train_datasets/MS_COCO_APGD_1/images`
  - `apgd_4_images.zip` → `./clip_train_datasets/MS_COCO_APGD_4/images`

#### 3. COCO 2017 Validation Set
- Download from [COCO website](https://cocodataset.org/#download)
- Copy images to all CLIP training dataset folders:
  - `./clip_train_datasets/MS_COCO/images`
  - `./clip_train_datasets/MS_COCO_APGD_4/images`
  - `./clip_train_datasets/MS_COCO_APGD_1/images`
  - `./clip_train_datasets/MS_COCO_COCO_CF/images`

#### 4. COCO Captions and Classification Datasets
- Download `ms_coco_captions.json` from [TU Berlin Cloud](https://tubcloud.tu-berlin.de/s/YdRcyp888N5qwkx)
- Place in: `./clip_train_datasets/MS_COCO`
- Download classification datasets from [TU Berlin Cloud](https://tubcloud.tu-berlin.de/s/YdRcyp888N5qwkx):
  - `Caltech101.zip` → unzip in `./image_classification_datasets`
  - `Caltech256.zip` → unzip in `./image_classification_datasets`
- For ImageNet: Download externally and set path in `vlm_eval/clip_classification.py` line 52

---

## Usage

### Sparse vs Non-Sparse Attacks Evaluation

Evaluate vision-language models against adversarial attacks. The following command demonstrates the evaluation setup (available in `bash/run_script.sh` and `bash/run_script_slurm.sh`):
```bash
python -m vlm_eval.run_evaluation \
--eval_flickr30 \
--dont_save_adv \
--verbose \
--attack saif --eps 255 --steps 100 --mask_out none --mu 1.5 --search_steps 2 --lam 0.005 --k 1000 \
--pert_factor_graph 0 \
--itr 0 \
--itr_clip 0 \
--itr_dataset base \
--itr_method APGD_1 \
--vision_encoder_pretrained openai \
--num_samples 8 \
--trial_seeds 42 \
--num_trials 1 \
--shots 0 \
--batch_size 1 \
--results_file  res9B \
--model open_flamingo \
--out_base_path /PATH/TO/Robust_mmfm/Results/open_flamingo \
--vision_encoder_path ViT-L-14 \
--checkpoint_path /PATH/TO/HUGGINGFACE/hub/models--openflamingo--OpenFlamingo-9B-vitl-mpt7b/snapshots/7e36809c73d038829ad5fba9d0cc949b4e180562/checkpoint.pt \
--lm_path anas-awadalla/mpt-7b \
--lm_tokenizer_path anas-awadalla/mpt-7b \
--precision float16 \
--cross_attn_every_n_layers 4 \
--coco_train_image_dir_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/COCO/train2014 \
--coco_val_image_dir_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/COCO/val2014 \
--coco_karpathy_json_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/COCO/karpathy_coco.json \
--coco_annotations_json_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/COCO/captions_val2014.json \
--coco_cf_image_dir_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/COCO_CF \
--flickr_image_dir_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/Flickr30k/Images \
--flickr_karpathy_json_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/Flickr30k/karpathy_flickr30k.json \
--flickr_annotations_json_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/Flickr30k/dataset_flickr30k_coco_style.json \
--vizwiz_train_image_dir_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/VizWiz/train \
--vizwiz_test_image_dir_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/VizWiz/val \
--vizwiz_train_questions_json_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/VizWiz/train_questions_vqa_format.json \
--vizwiz_train_annotations_json_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/VizWiz/train_annotations_vqa_format.json \
--vizwiz_test_questions_json_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/VizWiz/val_questions_vqa_format.json \
--vizwiz_test_annotations_json_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/VizWiz/val_annotations_vqa_format.json \
--vqav2_train_image_dir_path /home/htc/kchitranshi/SCRATCH/COCO/train2014 \
--vqav2_train_questions_json_path /home/htc/kchitranshi/SCRATCH/vqav2/v2_OpenEnded_mscoco_train2014_questions.json \
--vqav2_train_annotations_json_path /home/htc/kchitranshi/SCRATCH/vqav2/v2_mscoco_train2014_annotations.json \
--vqav2_test_image_dir_path /home/htc/kchitranshi/SCRATCH/COCO/val2014 \
--vqav2_test_questions_json_path /home/htc/kchitranshi/SCRATCH/vqav2/v2_OpenEnded_mscoco_val2014_questions.json \
--vqav2_test_annotations_json_path /home/htc/kchitranshi/SCRATCH/vqav2/v2_mscoco_val2014_annotations.json \
--textvqa_image_dir_path /mnt/datasets/textvqa/train_images \
--textvqa_train_questions_json_path /home/htc/kchitranshi/SCRATCH/RobustVLM/textvqa/train_questions_vqa_format.json \
--textvqa_train_annotations_json_path /home/htc/kchitranshi/SCRATCH/RobustVLM/textvqa/train_annotations_vqa_format.json \
--textvqa_test_questions_json_path /home/htc/kchitranshi/SCRATCH/RobustVLM/textvqa/val_questions_vqa_format.json \
--textvqa_test_annotations_json_path /home/htc/kchitranshi/RobustVLM/textvqa/val_annotations_vqa_format.json \
--ok_vqa_train_image_dir_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/COCO/train2014 \
--ok_vqa_train_questions_json_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/OKVQA/OpenEnded_mscoco_train2014_questions.json \
--ok_vqa_train_annotations_json_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/OKVQA/mscoco_train2014_annotations.json \
--ok_vqa_test_image_dir_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/COCO/val2014 \
--ok_vqa_test_questions_json_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/OKVQA/OpenEnded_mscoco_val2014_questions.json \
--ok_vqa_test_annotations_json_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/OKVQA/mscoco_val2014_annotations.json
```

#### Configuration Options

**Attack Types:**
- APGD attack: `--attack apgd --eps <epsilon>`
- SAIF attack: `--attack saif --eps <epsilon> --k <k_value>`
- No attack (clean): `--attack none`
- Targeted attack (COCO only): `--targeted --target_str "TARGET_STRING"`

**Shot Settings:**
- 0-shot: `--shots 0`
- 4-shot: `--shots 4`
  - Query mode: `--mask_out context`
  - All mode: `--mask_out none`

**Evaluation Tasks:**
- Image Captioning:
  - COCO: `--eval_coco`
  - Flickr30k: `--eval_flickr30`
- Visual Question Answering:
  - VizWiz: `--eval_vizwiz`
  - OK-VQA: `--eval_ok_vqa`

**Other Options:**
- Save adversarial samples as `.pt` files: remove `--dont_save_adv`
- Generate perturbation factor graph (0-shot only): `--pert_factor_graph 1`

#### Running the Scripts

```bash
# Make scripts executable
chmod +x ./bash/run_script.sh
chmod +x ./bash/run_script_slurm.sh

# Run locally or remotely
./bash/run_script.sh

# Run on SLURM cluster
sbatch ./bash/run_script_slurm.sh
```

### Fine-tuning CLIP Models

Fine-tune CLIP models on adversarial examples (APGD) and COCO counterfactuals. Example command (available in `bash/train_clip.sh` and `bash/train_clip_slurm.sh`):

```bash
python vlm_eval/clip_train.py \
    --num_epochs 20 \
    --data_seeds 112 113 114 115 \
    --data_name base \
    --method APGD_4 \
    --batch_size 128 \
    --learning_rate 5e-7 \
    --save_model \
    --save_model_path ./fine_tuned_clip_models/APGD_4/
```

This fine-tunes CLIP for 20 epochs on the `base` dataset with APGD attack (ε=4/255).

#### Parameters

- `--data_name`: Dataset size variant
  - `MS_COCO`: Standard MS COCO (see thesis appendix)
  - `base`: Base subset
  - `medium`: Medium subset
  - `all`: Complete dataset

- `--method`: Training method
  - `APGD_4`: APGD with ε=4/255
  - `APGD_1`: APGD with ε=1/255
  - `COCO_CF`: COCO Counterfactuals
  - `NONE`: Clean MS COCO (no perturbations)

- `--data_seeds`: Random seeds for dataset sampling (e.g., `112 113 114 115`)

#### Running the Scripts

```bash
# Make scripts executable
chmod +x ./bash/clip_train.sh
chmod +x ./bash/clip_train_slurm.sh

# Run locally or remotely
./bash/clip_train.sh

# Run on SLURM cluster
sbatch ./bash/clip_train_slurm.sh
```

### Zero-Shot Image Classification

Evaluate fine-tuned CLIP models on image classification tasks. Example command (available in `bash/clip_classification.sh` and `bash/clip_classification_slurm.sh`):

```bash
python vlm_eval/clip_classification.py \
    --data base \
    --method COCO_CF \
    --dataset Caltech101
```

This performs zero-shot classification on Caltech101 using a CLIP model fine-tuned on the `base` COCO counterfactuals dataset.

#### Parameters

- `--data`: Dataset variant
  - `MS_COCO`, `base`, `medium`, `all`: Fine-tuned models
  - `non_fine_tuned`: Pre-trained CLIP only (no fine-tuning)

- `--method`: `APGD_4`, `APGD_1`, `COCO_CF`, `NONE`

- `--dataset`: Classification dataset
  - `Food101`, `CIFAR10`, `CIFAR100`, `ImageNet`, `Caltech101`, `Caltech256`

**Note:** Evaluation is hardcoded to 20 epochs.

#### Running the Scripts

```bash
chmod +x ./bash/clip_classification.sh
chmod +x ./bash/clip_classification_slurm.sh

# Run locally or remotely
./bash/clip_classification.sh

# Run on SLURM cluster
sbatch ./bash/clip_classification_slurm.sh
```

---

### Image-Text Retrieval

Perform image-to-text (i2t) and text-to-image (t2i) retrieval tasks:
```bash
python -m vlm_eval.run_evaluation \
--eval_flickr30 \
--dont_save_adv \
--verbose \
--attack none --eps 255 --steps 100 --mask_out none --mu 1.5 --search_steps 2 --lam 0.005 --k 1000 \
--pert_factor_graph 0 \
--itr 1 \
--itr_clip 0 \
--itr_dataset base \
--itr_method APGD_1 \
--vision_encoder_pretrained openai \
--num_samples 1000 \
--trial_seeds 42 \
--num_trials 1 \
--shots 0 \
--batch_size 1 \
--results_file  res9B \
--model open_flamingo \
--out_base_path /PATH/TO/Robust_mmfm/Results/open_flamingo \
--vision_encoder_path ViT-L-14 \
--checkpoint_path /PATH/TO/HUGGINGFACE/hub/models--openflamingo--OpenFlamingo-9B-vitl-mpt7b/snapshots/7e36809c73d038829ad5fba9d0cc949b4e180562/checkpoint.pt \
--lm_path anas-awadalla/mpt-7b \
--lm_tokenizer_path anas-awadalla/mpt-7b \
--precision float16 \
--cross_attn_every_n_layers 4 \
--coco_train_image_dir_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/COCO/train2014 \
--coco_val_image_dir_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/COCO/val2014 \
--coco_karpathy_json_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/COCO/karpathy_coco.json \
--coco_annotations_json_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/COCO/captions_val2014.json \
--coco_cf_image_dir_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/COCO_CF \
--flickr_image_dir_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/Flickr30k/Images \
--flickr_karpathy_json_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/Flickr30k/karpathy_flickr30k.json \
--flickr_annotations_json_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/Flickr30k/dataset_flickr30k_coco_style.json \
--vizwiz_train_image_dir_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/VizWiz/train \
--vizwiz_test_image_dir_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/VizWiz/val \
--vizwiz_train_questions_json_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/VizWiz/train_questions_vqa_format.json \
--vizwiz_train_annotations_json_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/VizWiz/train_annotations_vqa_format.json \
--vizwiz_test_questions_json_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/VizWiz/val_questions_vqa_format.json \
--vizwiz_test_annotations_json_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/VizWiz/val_annotations_vqa_format.json \
--vqav2_train_image_dir_path /home/htc/kchitranshi/SCRATCH/COCO/train2014 \
--vqav2_train_questions_json_path /home/htc/kchitranshi/SCRATCH/vqav2/v2_OpenEnded_mscoco_train2014_questions.json \
--vqav2_train_annotations_json_path /home/htc/kchitranshi/SCRATCH/vqav2/v2_mscoco_train2014_annotations.json \
--vqav2_test_image_dir_path /home/htc/kchitranshi/SCRATCH/COCO/val2014 \
--vqav2_test_questions_json_path /home/htc/kchitranshi/SCRATCH/vqav2/v2_OpenEnded_mscoco_val2014_questions.json \
--vqav2_test_annotations_json_path /home/htc/kchitranshi/SCRATCH/vqav2/v2_mscoco_val2014_annotations.json \
--textvqa_image_dir_path /mnt/datasets/textvqa/train_images \
--textvqa_train_questions_json_path /home/htc/kchitranshi/SCRATCH/RobustVLM/textvqa/train_questions_vqa_format.json \
--textvqa_train_annotations_json_path /home/htc/kchitranshi/SCRATCH/RobustVLM/textvqa/train_annotations_vqa_format.json \
--textvqa_test_questions_json_path /home/htc/kchitranshi/SCRATCH/RobustVLM/textvqa/val_questions_vqa_format.json \
--textvqa_test_annotations_json_path /home/htc/kchitranshi/RobustVLM/textvqa/val_annotations_vqa_format.json \
--ok_vqa_train_image_dir_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/COCO/train2014 \
--ok_vqa_train_questions_json_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/OKVQA/OpenEnded_mscoco_train2014_questions.json \
--ok_vqa_train_annotations_json_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/OKVQA/mscoco_train2014_annotations.json \
--ok_vqa_test_image_dir_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/COCO/val2014 \
--ok_vqa_test_questions_json_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/OKVQA/OpenEnded_mscoco_val2014_questions.json \
--ok_vqa_test_annotations_json_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/OKVQA/mscoco_val2014_annotations.json
```

This evaluates i2t and t2i on the Flickr30k 1K test set (1000 samples) using a CLIP model fine-tuned on the `base` APGD dataset (ε=1/255).

#### Parameters

- `--itr_dataset`: Dataset for fine-tuned CLIP model
  - `MS_COCO`, `base`, `medium`, `all`: Fine-tuned variants
  - `non_fine_tuned`: Pre-trained CLIP only

**Note:** Image-text retrieval does not support targeted attacks or 4-shot settings.

---



## License

Please refer to the original [RobustVLM repository](https://github.com/chs20/RobustVLM) for licensing information.

## Acknowledgments

This code is adapted from the [RobustVLM](https://github.com/chs20/RobustVLM) repository. We thank the original authors for their foundational work.
