import json
import torch
import numpy as np
import random



def main():

    # Intialising seeds for data
    data_seeds = [i for i in range(107,122)]

    ms_coco_base_anno_path = "./clip_train_datasets/MS_COCO/ms_coco_captions.json"
    attack_base_anno_path = "./clip_train_datasets/COCO_CF/examples.jsonl"

    data_names = ["base","medium","all"]

    ms_coco_array = []
    attack_array = []

    with open(ms_coco_base_anno_path, 'r') as file:
        for line in file:
            ms_coco_array.append(json.loads(line))


    with open(attack_base_anno_path, 'r') as file:
        for line in file:
            attack_array.append(json.loads(line))

    for data_name in data_names:
        for data_seed in data_seeds:
            if data_name == "base":
                num_ms_coco_samples = 8705
                num_attacks_samples = 4353 # These many pairs of samples with their counterfactuals or adv attack samples. Effectively 8706 in total.
            elif data_name == "medium":
                num_ms_coco_samples = 17410
                num_attacks_samples = int(0.75 * 17410) # These many pairs of samples with their counterfactuals or adv attack samples. Effectively 26115 in total.
            elif data_name == "all":
                num_ms_coco_samples = 17410
                num_attacks_samples = 17410 # These many pairs of samples with their counterfactuals or adv attack samples. Effectively 34820 in total.

            np.random.seed(data_seed)
            ms_coco_rand_indices = np.random.choice(len(ms_coco_array), num_ms_coco_samples, replace=False)
            attack_rand_indices = np.random.choice(len(attack_array), num_attacks_samples, replace=False)

            ms_coco_samples = [ms_coco_array[int(i)] for i in ms_coco_rand_indices]
            attack_samples = [attack_array[int(i)] for i in attack_rand_indices]
            attack_samples = [{"image_id": batch["id"], "image_name": batch[f"image_{i}"] + ".jpg", "caption": batch[f"caption_{i}"]} for batch in attack_samples for i in range(2)]

            random.seed(42)
            combined_dataset = ms_coco_samples + attack_samples

            random.shuffle(combined_dataset)

            if data_name != 'all':
                with open(f"./clip_train_datasets/MS_COCO_APGD_4/json_files/data_name_{data_name}_data_seed_{data_seed}.json", 'w') as file:
                    for line in combined_dataset:
                        file.write(json.dumps(line) + '\n')
            else:
                with open(f"./clip_train_datasets/MS_COCO_APGD_4/json_files/data_name_{data_name}.json", 'w') as file:
                    for line in combined_dataset:
                        file.write(json.dumps(line) + '\n')

if __name__ == "__main__":
    main()
