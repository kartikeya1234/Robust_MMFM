import os
import json
import torchvision.datasets as dset
import torchvision.transforms as transforms
from coco_cf import COCO_CF_dataset
from torch.utils.data import DataLoader

def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch

coco_2017 = dset.CocoCaptions(root='./open_flamingo_datasets/COCO_2017/val2017/',
                              annFile='./open_flamingo_datasets/COCO_2017/captions_val2017.json',
                              transform=transforms.ToTensor())

coco_cf = COCO_CF_dataset(base_dir='./open_flamingo_datasets/COCO_CF/')
dl_coco_cf = DataLoader(coco_cf, batch_size=100,collate_fn=custom_collate_fn)


# Collect both captions from each batch in one step
coco_cf_captions = []

for batch in dl_coco_cf:
    # Extend the list with both captions at once without list comprehension
    coco_cf_captions.extend([caption.replace('.','').replace(",","").replace("-"," ").replace("'s","").lower().strip() for caption in batch['caption_0']])

ms_coco_gen_indices = []
coco_cf_captions_set = set(coco_cf_captions)

for index in range(len(coco_2017)):
    image_id = coco_2017.ids[index]
    _,captions = coco_2017[index]


    matches = [s for s in captions if s.replace(".","").replace(",","").replace("'s","").replace("-"," ").lower().strip() in coco_cf_captions_set]


    for match in matches:
        ms_coco_gen_indices.append((image_id,match))
ms_coco_gen_indices = ms_coco_gen_indices[:17410]
print(ms_coco_gen_indices)
ms_coco = [{'image_id': image_index,'caption': caption} for (image_index, caption) in ms_coco_gen_indices]

file_path = 'ms_coco_captions.json'

# Save the dictionary to a JSON file

import os

# Base path where the images are located
base_image_path = '/home/kc/Downloads/val2017/'

# Assuming ms_coco_gen_indices is a list of (image_index, caption) tuples
ms_coco_gen_indices = [(image_index, caption) for (image_index, caption) in ms_coco_gen_indices]

# List to store the updated entries with pathtoimage included
updated_ms_coco_gen_indices = []

# Process each (image_index, caption) in ms_coco_gen_indices
for image_index, caption in ms_coco_gen_indices:
    # Construct the full path to the image file based on the image_index
    pathtoimage = f"{image_index:012d}.jpg"  # Ensure image_index is 12 digits with padding

    # Append the new entry as (image_index, pathtoimage, caption)
    updated_ms_coco_gen_indices.append((image_index, pathtoimage, caption))

# Now ms_coco_gen_indices includes (image_index, pathtoimage, caption)
ms_coco_gen_indices = updated_ms_coco_gen_indices
ms_coco = [{'image_id': image_index,'image_name': image_name,'caption': caption} for (image_index,image_name ,caption) in ms_coco_gen_indices]

with open(file_path, 'w') as json_file:
    for row in ms_coco:
        json.dump(row, json_file)
        json_file.write('\n')
