from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import json
from PIL import Image


class MS_COCO_dataset(Dataset):

    def __init__(self, base_dir, annotation_file=None):

        self.data= []
        self.img_dir = base_dir + '/images'
        self.annotation_file = base_dir + annotation_file

        with open(self.annotation_file, 'r') as file:
            for line in file:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract the relevant info from the JSONL entry
        img_name = os.path.join(self.img_dir, f"{self.data[idx]['image_name']}")
        caption = self.data[idx]['caption']
        sample_id = self.data[idx]['image_id']

        # Load the image using PIL
        img = Image.open(img_name)

        return {"id": sample_id, 
                "image": img,
                "caption": caption
                }

class COCO_CF_dataset(Dataset):

    def __init__(self, base_dir):

        self.data= []
        self.img_dir = base_dir + '/images'
        self.annotation_file = base_dir + "/examples.jsonl"

        with open(self.annotation_file, 'r') as file:
            for line in file:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract the relevant info from the JSONL entry
        img_0_name = os.path.join(self.img_dir, f"{self.data[idx]['image_0']}.jpg")
        img_1_name = os.path.join(self.img_dir, f"{self.data[idx]['image_1']}.jpg")

        caption_0 = self.data[idx]['caption_0']
        caption_1 = self.data[idx]['caption_1']
        sample_id = self.data[idx]['id']

        # Load the image using PIL
        img_0 = Image.open(img_0_name)
        img_1 = Image.open(img_1_name)

        return {"id": sample_id, 
                "caption_0": caption_0, 
                "caption_1": caption_1, 
                "image_0": img_0, 
                "image_1": img_1}

def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch

if __name__ == "__main__":

    base_dir = '/home/htc/kchitranshi/SCRATCH/MS_COCO/'   
    data = MS_COCO_dataset(base_dir=base_dir)
    data_loader = DataLoader(data, batch_size=10,collate_fn=custom_collate_fn)
    
    for batch in data_loader:
        print(batch)
        break




    