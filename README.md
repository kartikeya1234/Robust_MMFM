# Robustness of Multi-Modal Foundational models

#### Prerequisites
- Execute `$ cd Robust_mmfm` command to get into the said directory. **Ensure** that Python version is `3.11.*`. Execute `$ pip install -r requirements.txt` to download all the required packages. 
##### Sparse vs Non-Sparse attacks evaluation
1. OpenFlamingo 9 billion parameters model can be downloaded from [here](https://huggingface.co/openflamingo/OpenFlamingo-9B-vitl-mpt7b) by following the provided instructions. After downloading it, it should be located in `/HOME/.cache/huggingface/hub/` folder with the name `models--openflamingo--OpenFlamingo-9B-vitl-mpt7b`. 
2. Install `jdk1.8.0_202` from [here](https://tubcloud.tu-berlin.de/s/YdRcyp888N5qwkx) as it is required for the computation of CIDEr score. You might have to add the line `export PATH=$PATH:JDKDir/jdk1.8.0_202/bin` in your `.bashrc` or  `.zshrc` file. Also **set** `$ LANG=en_US.UTF-8` if the `LANG` environment variable is in Deutsch. You can check it by typing `$ echo $LANG`.
3. Download `VizWiz` dataset (train and validation) from [here](https://vizwiz.org/tasks-and-datasets/vqa/). The files such as `train_questions_vqa_format.json`, `train_annotations_vqa_format.json`, `val_questions_vqa_format.json` and `val_annotations_vqa_format.json` are already there in the respective folder, however, if they are corrupted then download them from [here](https://vizwiz.org/tasks-and-datasets/vqa/). Copy all of the training and val images into the `./open_flamingo_datasets/VizWiz/train` and `./open_flamingo_datasets/VizWiz/val` folder, respectively.
4. The `OKVQA` dataset (Training and Testing images) from [here](https://okvqa.allenai.org/download.html). The files such as `OpenEnded_mscoco_train2014_questions.json`, `mscoco_train2014_annotations.json`, `OpenEnded_mscoco_val2014_questions.json` and `mscoco_val2014_annotations.json` are already there in the respective folder, however, if they are corrupted then download them from [here](https://okvqa.allenai.org/download.html). Copy all of them into the `./open_flamingo_datasets/OKVQA` folder.
5. Download the `Flickr30k` dataset from [here](https://github.com/awsaf49/flickr-dataset) by following the instructions given on the GitHub page. Files such as `karpathy_flickr30k.json` and `dataset_flickr30k_coco_style.json` are already there in the respective folder, however, if they are corrupted then download them from [here](https://nc.mlcloud.uni-tuebingen.de/index.php/s/mtRnQFaZJkR9zaX). Copy all the images into the `./open_flamingo_datasets/Flickr30k/Images` folder.
6. Download the `COCO` dataset (Training and Validation sets) 2014 from [here](https://cocodataset.org/#download). Files such as `karpathy_coco.json` (from ) and `captions_val2014.json` are already there in their respective places, however, if they are corrupted then download from [here](https://nc.mlcloud.uni-tuebingen.de/index.php/s/mtRnQFaZJkR9zaX) and [here](https://github.com/tylin/coco-caption/blob/master/annotations/captions_val2014.json), respectively. Copy all the training and val images to `./open_flamingo_datasets/COCO/train2014` and `./open_flamingo_datasets/COCO/val2014` folder.
##### Fine-tuning CLIP models (APGD vs COCO CFs)
1. Download the `images.zip` for `COCO-CFs` from [here](https://huggingface.co/datasets/Intel/COCO-Counterfactuals/tree/main/data). Unzip and put the images in `./open_flamingo_datasets/COCO_CF/images` and `./clip_train_datasets/MS_COCO_COCO_CF/images`, and copy all the original images (all images ending with `_0.jpg`, like  `236308_0_img_0.jpg`) into the `./clip_train_datasets/MS_COCO_APGD_4/images`, `./clip_train_datasets/MS_COCO_APGD_1/images`. Use the command `$ cp ./open_flamingo_datasets/COCO_CF/images/*_0.jpg ./clip_train_datasets/MS_COCO_APGD_4/images` for transfering it into `./clip_train_datasets/MS_COCO_APGD_4`. Do the same for all.
2. Download the `apgd_1_images.zip` and `apgd_4_images.zip` from [here](https://tubcloud.tu-berlin.de/s/YdRcyp888N5qwkx) and [here](https://tubcloud.tu-berlin.de/s/YdRcyp888N5qwkx), respectively. Copy all of the images in them into the `./clip_train_datasets/MS_COCO_APGD_1/images` and `./clip_train_datasets/MS_COCO_APGD_4/images` folders, respectively.
3. Download the `COCO` 2017 validation set from [here](https://cocodataset.org/#download). Copy all the images in the folder into the `./clip_train_datasets/MS_COCO/images`, `./clip_train_datasets/MS_COCO_APGD_4/images`,`./clip_train_datasets/MS_COCO_APGD_1/images` and `./clip_train_datasets/MS_COCO_COCO_CF/images` folders.
4. Download the `ms_coco_captions.json` file from [here](https://tubcloud.tu-berlin.de/s/YdRcyp888N5qwkx). Place it into the `./clip_train_datasets/MS_COCO` folder.
5. Download the `Caltech101` and `Caltech256` datasets in `.zip` format from [here](https://tubcloud.tu-berlin.de/s/YdRcyp888N5qwkx). Unzip them in the `./image_classification_datasets` folder. For the `ImageNet` dataset, the dataset would have to be downloaded externally, and then the path must be filled in the `clip_classification.py` file in the `./vlm_eval` folder, line 52 `imagenet_path=''`
----
----
#### Experiments

##### Sparse vs Non-Sparse attacks evaluation
- The command for executing the sparse vs non-sparse attacks experiments is given below. Something similar should be given in both of the files, `run_script.sh` and `run_script_slurm.sh`, which are located in the `bash` folder.
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
--ok_vqa_test_annotations_json_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/OKVQA/mscoco_val2014_annotations.json \
```
- For targeted attacks, set `--targeted --target_str "TARGET_STRING"`, otherwise remove it. Targeted attacks are **only available** for `COCO` dataset.
- For the 4-shot sub-setting, set `--shots 4`. For `query` mode, set `--mask_out context`. For `all` mode, set `--maskout none`. For the 0-shot sub-setting, set `--shots 0`.
- For switching adversarial attacks, set `--attack apgd` for the APGD attack and `--attack saif` for the SAIF attack. The hyper-parameters can be manipulated for APGD attack by changing `--eps`, and for SAIF, `--eps` and `--k`. For unperturbed results, set `--attack none`. To save the adversarial samples as Pytorch tensors `.pt`, remove `--dont_save_adv`.
- To conduct image captioning with COCO dataset, set to `--eval_coco`, and for Flickr30k, set to `--eval_flickr30`. For VQA tasks, set to `--eval_vizwiz` for VizWiz dataset and `--eval_ok_vqa` for OkVQA dataset.
- To reproduce the perturbation factor graph, set `--pert_factor_graph 1`. It **only works** in the 0-shot sub-setting.
- To execute the files, follow these commands
```bash
$ chmod +x ./bash/run_script.sh
$ chmod +x ./bash/run_script_slurm.sh
$ ./bash/run_script.sh # For running it on your local machine or remotely OR
$ sbatch ./bash/run_script_slurm.sh # For running it on SLURM (https://slurm.schedmd.com/documentation.html)
```
----
##### Fine-tuning CLIP models (APGD vs COCO CFs)
- Given below is the command for fine-tuning CLIP pre-trained models. It is available in the `train_clip.sh` and `train_clip_slurm.sh` files.
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
- This command will fine-tune pre-trained CLIP models for 20 epochs on the `base` dataset for the APGD attack with $\epsilon=\frac{4}{255}$.
- To execute the files, follow these commands
```bash
$ chmod +x ./bash/clip_train.sh
$ chmod +x ./bash/clip_train_slurm.sh
$ ./bash/clip_train.sh # For running it on your local machine or remotely OR
$ sbatch ./bash/clip_train_slurm.sh # For running it on SLURM (https://slurm.schedmd.com/documentation.html)
```
- `--data_name` chooses which dataset to fine-tune the model on, while `--data_seeds` provides the seed for which randomly chosen `base`or `medium` dataset to choose from. 
- There are four options given for fine-tuning the model on for the APGD and Coco counterfactuals datasets - `[MS_COCO, base, medium, all]`. `MS_COCO` will train the model on MS COCO dataset that we derived and described in the thesis (in the Appendix), whereas `base, medium` or `all` will train them on the base, medium and all dataset of the given method.
- There are 4 methods available - `APGD_4, APGD_1`, `NONE` and `COCO_CF`. Choosing any one of them would train the model on the given dataset, except for `NONE` as it is used with `MS_COCO` as there is no adversarial sample or counterfactual present.

###### 0-Shot Image Classification
- Given below is the command for conducting the 0-shot image classification. It is given in the `clip_classification.sh` and `clip_classification_slurm.sh` files
```bash
python vlm_eval/clip_classification.py \
    --data base \
    --method COCO_CF \
    --dataset Caltech101
```

- This command will conduct 0-shot image classification on the pre-trained CLIP model fine-tuned on the `base` dataset of the COCO counterfactuals for the validation set of the dataset `Caltech101`.
- To execute the file, follow these commands
```bash
$ chmod +x ./bash/clip_classification.sh
$ chmod +x ./bash/clip_classification_slurm.sh
$ ./bash/clip_classificationrun_script.sh # For running it on your local machine or remotely OR
$ sbatch ./bash/clip_classificationrun_script_slurm.sh # For running it on SLURM (https://slurm.schedmd.com/documentation.html)
```
- Availble datasets are `Food101, CIFAR10, CIFAR100, ImageNet,  Caltech101` and `Caltech256`.  
- Again, the available options for `--data` are `MS_COCO, base, medium, all` and `non_fine_tuned`.
- To get 0-shot image classification accuracy for only pre-trained CLIP model, set `--data non_fine_tuned`, regardless of the method.
- It is **hardcoded** to 20 epochs.
###### Image-Text Retrieval
- Given below is the command to perform **i2t** and **t2i**.
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
--ok_vqa_test_annotations_json_path /PATH/TO/Robust_mmfm/open_flamingo_datasets/OKVQA/mscoco_val2014_annotations.json \
```
- This will conduct i2t and t2i on the flickr30k 1K test set for 1000 samples. The fine-tuned CLIP model in this case will be the one fine-tuned on `base` dataset of APGD with $\epsilon=\frac{1}{255}$ for all the data seeds.
- The `--itr_dataset` can be changed to `MS_COCO, base, medium, all` and `non_fine_tuned`. Again, to get i2t and t2i for a non fine-tuned CLIP model, set `--itr_dataset non_fine_tuned`.
- This will **not** work for targeted attack or 4-shots sub-setting.