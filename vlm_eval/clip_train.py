# Code adapted from https://github.com/ylaxor/clip-like/blob/main/fine-tune-clip.ipynb

from random import seed, shuffle
from typing import Callable
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from timm.scheduler import CosineLRScheduler



class ModelTrainer:

    def __init__(self,
                 model: Callable,
                 processor: Callable,
                 data_name: str,
                 train_data_loader: torch.utils.data.DataLoader,
                 val_data_loader: torch.utils.data.DataLoader,
                 num_epochs: int,
                 learning_rate: float = 5e-5,
                 weight_decay: float = 1e-3,
                 device: str = "cuda:0",
                 save_model: bool = False,
                 save_model_path: str = "./fine_tuned_clip_models",
                 data_seed: int = 42,
                 method="COCO_CF",
    ) -> None:

        self.model = model
        self.processor = processor
        self.data_name = data_name
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.save_model = save_model
        self.save_model_path = save_model_path
        self.data_seed = data_seed
        self.method = method

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )


    def train(self):
        self.model.train()
        lr_scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=self.num_epochs * len(self.train_data_loader),
            lr_min=2e-7,
            warmup_lr_init=1e-7,
            warmup_prefix=True,
            warmup_t=3,
            cycle_limit=1,
            t_in_epochs=False,
        )
        progress_bar = tqdm(range(self.num_epochs))
        for epoch in progress_bar:
            running_loss = 0.0
            for batch_idx, batch in enumerate(self.train_data_loader):
                self.optimizer.zero_grad()
                processed_input = self.processor(text=batch["caption"],
                                                 images=batch["image"],
                                                 return_tensors="pt",
                                                 padding=True,
                                                 max_length=128,
                                                 truncation=True
                )
                outputs = self.model(input_ids=processed_input['input_ids'].squeeze().to(self.device),
                                     pixel_values=processed_input['pixel_values'].squeeze().to(self.device),
                                     attention_mask=processed_input['attention_mask'].squeeze().to(self.device),
                                     return_loss=True
                )
                loss = outputs.loss
                loss.backward()
                running_loss += loss.item() * len(batch["caption"])
                self.optimizer.step()
                lr_scheduler.step_update(batch_idx + epoch * len(self.train_data_loader))

            print(f"Epoch {epoch+1}/{self.num_epochs} Loss: {running_loss/len(self.train_data_loader.dataset):.4f}")
            progress_bar.set_postfix(
                epoch="{}/{}".format(epoch+1,self.num_epochs),
                loss=running_loss/len(self.train_data_loader.dataset),
                lr=self.optimizer.param_groups[0]["lr"]
            )

        if self.save_model:
            if self.data_name not in ['MS_COCO','all']:
                torch.save(self.model.state_dict(), self.save_model_path + f'clip_model_dataset_{self.data_name}_method_{self.method}_num_epochs_{self.num_epochs}_data_seed_{self.data_seed}.pt')
                print(f"Saving fine-tuned model as clip_model_dataset_{self.data_name}_method_{self.method}_num_epochs_{self.num_epochs}_data_seed_{self.data_seed}.pt")
            else:
                torch.save(self.model.state_dict(), self.save_model_path + f'clip_model_dataset_{self.data_name}_method_{self.method}_num_epochs_{self.num_epochs}.pt')
                print(f"Saving fine-tuned model as clip_model_dataset_{self.data_name}_method_{self.method}_num_epochs_{self.num_epochs}.pt")

    def eval(self):
        self.model.eval()
        nb_batches = len(self.val_data_loader)
        tqdm_object = tqdm(self.val_data_loader, total=len(self.val_data_loader))
        epoch_loss = 0.0
        for i, batch in enumerate(tqdm_object):
            processed_input = self.processor(text=batch["caption"],
                                                 images=batch["image"],
                                                 return_tensors="pt",
                                                 padding=True,
                                                 max_length=128,
                                                 truncation=True
                )
            outputs = self.model(
                input_ids=processed_input['input_ids'].squeeze().to(self.device),
                attention_mask=processed_input['attention_mask'].squeeze().to(self.device),
                pixel_values=processed_input['pixel_values'].squeeze().to(self.device),
                return_loss=True)
            loss, logits_per_image = outputs.loss, outputs.logits_per_image
            epoch_loss += loss.item()
            tqdm_object.set_postfix(
                batch="{}/{}".format(i+1,nb_batches),
                dev_loss=loss.item(),
                )
        epoch_loss = epoch_loss / nb_batches
        print(f"Eval loss: {epoch_loss}")

def main():
    import os
    #os.environ['HF_HOME'] = ''    Add path for saved hugging face models

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--data_name', type=str, default="MS_COCO", choices=["MS_COCO","base","medium","all"])
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--method', type=str, choices=['COCO_CF','APGD_1','APGD_4','NONE'])
    parser.add_argument('--save_model_path', type=str, default="./fine_tuned_clip_models")
    parser.add_argument(
        "--data_seeds",
        nargs="+",
        type=int,
        default=[107],
        help="Seeds to use for each trial for picking demonstrations and eval sets",
    )
    args = parser.parse_args()
    if args.data_name == 'MS_COCO':
        assert args.data_name == 'MS_COCO' and args.method == 'NONE', "Only NONE method is allowed with MS_COCO dataset"

    from torch.utils.data import DataLoader
    from coco_cf_loader import MS_COCO_dataset, custom_collate_fn

    torch.manual_seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


    for data_seed in args.data_seeds:

        if args.data_name not in ['MS_COCO', 'all']:
            print(f"Data Seed: {data_seed} | Data Name: {args.data_name} | Method: {args.method}")
            dataset = MS_COCO_dataset(base_dir=f'./clip_train_datasets/MS_COCO_{args.method}',
                                      annotation_file=f'/json_files/data_name_{args.data_name}_data_seed_{data_seed}.json')
        elif args.data_name == 'all':
            print(f"Data Name: {args.data_name} | Method: {args.method}")
            dataset = MS_COCO_dataset(base_dir=f'./clip_train_datasets/MS_COCO_{args.method}',
                                      annotation_file=f'/json_files/data_name_{args.data_name}.json')
        else:
            print(f"Data Name: {args.data_name} | Method: {args.method}")
            dataset = MS_COCO_dataset(base_dir=f'./clip_train_datasets/MS_COCO',
                                      annotation_file=f'/ms_coco_captions.json')

        train_size = int(0.8 * len(dataset))  # 80% for training
        val_size = len(dataset) - train_size   # 20% for validation

        # Randomly split into training and validation datasets
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Optional: Create DataLoaders for each subset
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=custom_collate_fn,drop_last=True)

        trainer = ModelTrainer(model=model,
                            processor=processor,
                            data_name=args.data_name,
                            train_data_loader=train_loader,
                            val_data_loader=val_loader,
                            num_epochs=args.num_epochs,
                            learning_rate=args.learning_rate,
                            weight_decay=1e-3,
                            device=device,
                            data_seed=data_seed,
                            save_model=args.save_model,
                            save_model_path=args.save_model_path,
                            method=args.method,
        )

        trainer.train()
        trainer.eval()
        if args.data_name in ['MS_COCO','all']:
            break


if __name__ == "__main__":
    main()
