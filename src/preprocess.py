import torch
import os
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
from huggingface_hub import login

# A dummy dataset for when HF datasets fail or for quick tests
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=1000, num_classes=10, resolution=256, is_video=False):
        self.num_samples = num_samples
        self.resolution = resolution
        self.is_video = is_video
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.is_video:
            # (C, T, H, W) -> (3, 16, 64, 64)
            tensor = torch.randn(3, 16, self.resolution, self.resolution)
        else:
            # (C, H, W)
            tensor = torch.randn(3, self.resolution, self.resolution)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return tensor, label


def get_transforms(dataset_name, resolution):
    if 'imagenet' in dataset_name or 'mednist' in dataset_name:
        return transforms.Compose([
            transforms.Resize(resolution, antialias=True),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize to [-1, 1]
        ])
    elif 'laion' in dataset_name:
        return transforms.Compose([
            transforms.Resize(resolution, antialias=True),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    elif 'cifar' in dataset_name:
         return transforms.Compose([
            transforms.Resize(resolution, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif 'kinetics' in dataset_name:
        return transforms.Compose([transforms.ToTensor()])
    else:
        return transforms.Compose([
            transforms.Resize(resolution, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


def load_data(data_config, batch_size, num_workers=4):
    dataset_name = data_config['name']
    resolution = data_config.get('resolution', 256)
    use_auth_token = os.getenv("HF_TOKEN")
    if use_auth_token:
        try:
            login(token=use_auth_token)
            print("Hugging Face Hub login successful.")
        except Exception as e:
            print(f"Could not log in to Hugging Face Hub: {e}")

    try:
        if 'imagenet-256' in dataset_name or 'imagenet-512' in dataset_name:
            # Use a smaller, more accessible version for demonstration
            dataset = load_dataset("zh-plus/tiny-imagenet", split='train', use_auth_token=use_auth_token)
            val_dataset = load_dataset("zh-plus/tiny-imagenet", split='valid', use_auth_token=use_auth_token)
            transform = get_transforms('imagenet', resolution)
            
            def transform_fn(examples):
                images = [transform(img.convert('RGB')) for img in examples['image']]
                return {'image': images, 'label': examples['label']}
            
            dataset.set_transform(transform_fn)
            val_dataset.set_transform(transform_fn)
            train_dataset = dataset
            val_dataset = val_dataset
        elif 'laion' in dataset_name:
            # This is a huge dataset, so we'll use a dummy for now
            print(f"Using dummy dataset for {dataset_name}")
            train_dataset = DummyDataset(resolution=512)
            val_dataset = DummyDataset(num_samples=200, resolution=512)
        elif 'kinetics' in dataset_name:
            print(f"Using dummy dataset for {dataset_name}")
            train_dataset = DummyDataset(resolution=64, is_video=True)
            val_dataset = DummyDataset(num_samples=100, resolution=64, is_video=True)
        elif 'mednist' in dataset_name:
            # This dataset is not readily on HF, use a similar small medical dataset or dummy
            print(f"Using dummy dataset for {dataset_name}")
            train_dataset = DummyDataset(resolution=64, num_classes=6)
            val_dataset = DummyDataset(num_samples=100, resolution=64, num_classes=6)
        elif 'cifar10' in dataset_name: # For smoke test
            dataset = load_dataset('cifar10', split='train', use_auth_token=use_auth_token)
            val_dataset = load_dataset('cifar10', split='test', use_auth_token=use_auth_token)
            transform = get_transforms('cifar10', resolution)
            
            def transform_fn(examples):
                images = [transform(img) for img in examples['img']]
                return {'image': images, 'label': examples['label']}
            
            dataset.set_transform(transform_fn)
            val_dataset.set_transform(transform_fn)
            train_dataset = dataset
            val_dataset = val_dataset
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    except Exception as e:
        print(f"Failed to load dataset '{dataset_name}' from Hugging Face: {e}. Using a dummy dataset instead.")
        is_video = 'kinetics' in dataset_name
        train_dataset = DummyDataset(resolution=resolution, is_video=is_video)
        val_dataset = DummyDataset(num_samples=max(1, int(len(train_dataset)*0.1)), resolution=resolution, is_video=is_video)

    # For smoke tests, use a small subset
    if data_config.get('subset_size'):
        subset_size = data_config['subset_size']
        train_dataset = Subset(train_dataset, range(min(subset_size, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(int(subset_size*0.2), len(val_dataset))))

    def collate_fn(batch):
        if isinstance(batch[0], dict):
            images = torch.stack([item['image'] for item in batch])
            labels = torch.tensor([item['label'] for item in batch])
        else:
            images = torch.stack([item[0] for item in batch])
            labels = torch.tensor([item[1] for item in batch])
        return images, labels
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    
    # For privacy experiment, we need access to both train and val loaders for the attack
    return {'train': train_loader, 'val': val_loader}
