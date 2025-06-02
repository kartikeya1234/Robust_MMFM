# Code adapted from https://github.com/openai/CLIP/blob/main/
from transformers import CLIPProcessor, CLIPModel
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets_classes_templates import data_seeds
import numpy as np
from datetime import datetime

def zeroshot_classifier(classnames, templates, processor, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to('cuda')
            class_embeddings = model.get_text_features(text_inputs['input_ids']) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

def classification_collate_fn(batch):
    images, labels = zip(*batch)
    labels = torch.tensor(labels)
    return images, labels

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, choices=['non_fine_tuned','MS_COCO','medium','base','all'], help='Data on which clip was fine-tuned')
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["CIFAR10", "CIFAR100", "ImageNet", "Caltech101", "Caltech256", "Food101"])
    parser.add_argument("--method",type=str, default="COCO_CF", choices=['COCO_CF','APGD_1','APGD_4','NONE'])
    args = parser.parse_args()

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_filename = f'./Results/fine_tuned_clip/zeroshot_image_classification_results_{args.dataset}_{args.data}_{args.method}_{current_time}.txt'
    with open(results_filename, 'w') as f:
        f.write(f'Arguments: {args}\n\n')

    if args.data == 'MS_COCO':
        assert args.method == 'NONE' and args.data == 'MS_COCO', 'Use NONE for method for MS_COCO data'

    imagenet_path = '/software/ais2t/pytorch_datasets/imagenet/' # Fill the path for imagenet here

    if args.dataset == "CIFAR10":
        from datasets_classes_templates import CIFAR10_CLASSES_TEMPLATES as classes_templates
        from torchvision.datasets import CIFAR10
        data = CIFAR10(root='./image_classification_datasets/cifar10/', train=False, download=True)
    elif args.dataset == "CIFAR100":
        from datasets_classes_templates import CIFAR100_CLASSES_TEMPLATES as classes_templates
        from torchvision.datasets import CIFAR100
        data = CIFAR100(root='./image_classification_datasets/cifar100/', train=False, download=True)
    elif args.dataset == "ImageNet":
        from datasets_classes_templates import ImageNet_CLASSES_TEMPLATES as classes_templates
        from torchvision.datasets import ImageNet
        data = ImageNet(root=imagenet_path, split='val')
    elif args.dataset == "Caltech101":
        torch.manual_seed(42)
        from datasets_classes_templates import Caltech101_CLASSES_TEMPLATES as classes_templates
        from torchvision.datasets import Caltech101
        data = Caltech101(root='./image_classification_datasets/', download=False)
        train_size = int(0.8 * len(data))  # 80% for training
        val_size = len(data) - train_size
        _, data = torch.utils.data.random_split(data, [train_size, val_size])
    elif args.dataset == "Caltech256":
        torch.manual_seed(42)
        from datasets_classes_templates import Caltech256_CLASSES_TEMPLATES as classes_templates
        from torchvision.datasets import Caltech256
        data = Caltech256(root='./image_classification_datasets/', download=False)
        train_size = int(0.8 * len(data))  # 80% for training
        val_size = len(data) - train_size
        _, data = torch.utils.data.random_split(data, [train_size, val_size])
    elif args.dataset == "Food101":
        from datasets_classes_templates import Food101_CLASSES_TEMPLATES as classes_templates
        from torchvision.datasets import Food101
        data = Food101(root='./image_classification_datasets/food101/', download=True, split='test')
    else:
        raise NotImplementedError

    print(f'Conducting zero-shot image classification on {args.dataset}')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_base_path = './fine_tuned_clip_models'
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    top1_list = []
    for data_seed in data_seeds:
        print(f'Conducting zero-shot image classification on {args.data} with seed {data_seed} for the method {args.method}')
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        if args.data != 'non_fine_tuned':
            if args.method != 'NONE':
                if args.data not in ['all']:
                    model.load_state_dict(torch.load(f'{model_base_path}/{args.method}/clip_model_dataset_{args.data}_method_{args.method}_num_epochs_20_data_seed_{data_seed}.pt'))
                else:
                    model.load_state_dict(torch.load(f'{model_base_path}/{args.method}/clip_model_dataset_{args.data}_method_{args.method}_num_epochs_20.pt'))
            elif args.method == 'NONE' and args.data == 'MS_COCO':
                model.load_state_dict(torch.load(f'{model_base_path}/{args.method}/clip_model_dataset_{args.data}_method_{args.method}_num_epochs_20.pt'))

        model.eval()

        data_loader = DataLoader(data, batch_size=128, collate_fn=classification_collate_fn, shuffle=False)

        zeroshot_weights = zeroshot_classifier(classes_templates['classes'],
                                            classes_templates['templates'],
                                            processor,
                                            model
        )

        with torch.no_grad():
            top1, top5, n = 0., 0., 0.
            for i, (images, target) in enumerate(tqdm(data_loader)):
                target = target.to(device)
                images = list(images)

                images = processor(images=images, return_tensors="pt").to(device)

                # predict
                image_features = model.get_image_features(images['pixel_values']).to(device)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = 100. * image_features @ zeroshot_weights

                # measure accuracy
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                top1 += acc1
                top5 += acc5
                n += image_features.size(0)

        top1 = (top1 / n) * 100
        top5 = (top5 / n) * 100

        with open(results_filename, 'a') as f:
            f.write(f'Seed {data_seed}: Top-1 Accuracy: {top1:.2f}, Top-5 Accuracy: {top5:.2f}\n')

        top1_list.append(top1)

        print(f"Top-1 accuracy: {top1:.2f}")
        print(f"Top-5 accuracy: {top5:.2f}")
        print('-'*40)

        if args.method == 'NONE' or args.data in ['MS_COCO','all'] or args.data == 'non_fine_tuned':
            break
    top1 = np.asarray(top1_list)
    print(f'Mean of the top 1 accuracy is {np.mean(top1)}')
    print(f'Standard deviation of the top 1 accuracy is {np.std(top1)}')

    with open(results_filename, 'a') as f:
        f.write(f'\nMean Top-1 Accuracy: {np.mean(top1):.2f}\n')
        f.write(f'Standard Deviation of Top-1 Accuracy: {np.std(top1):.2f}\n')

if __name__ == "__main__":
    main()
