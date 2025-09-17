import torch
from torchvision import transforms
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PreprocessedIterableDataset(IterableDataset):
    def __init__(self, hf_dataset, transform):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __iter__(self):
        for item in self.hf_dataset:
            try:
                image = self.transform(item['image'].convert("RGB"))
                yield {'image': image, 'label': item.get('label', -1)}
            except Exception as e:
                # Skip corrupted images
                logging.warning(f"Skipping a corrupted image. Error: {e}")
                continue

def get_dataloaders(config, smoke_test=False):
    """Creates and returns data loaders for training and evaluation."""
    logging.info("Preparing dataloaders...")
    token = os.getenv('HF_TOKEN')

    # As per experimental design: center-crop 256, float32 [-1,1], random horizontal flip
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) # Maps to [-1, 1]
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_dataset_name = config['datasets']['train']
    
    try:
        # Use streaming to avoid downloading the entire huge dataset
        train_hf_dataset = load_dataset(train_dataset_name, split='train', streaming=True, use_auth_token=token)
        val_hf_dataset = load_dataset(config['datasets']['eval'], split='validation', streaming=True, use_auth_token=token)
        
        if smoke_test:
            train_hf_dataset = train_hf_dataset.take(100) # Small subset for smoke test
            val_hf_dataset = val_hf_dataset.take(20)

        train_dataset = PreprocessedIterableDataset(train_hf_dataset, train_transform)
        val_dataset = PreprocessedIterableDataset(val_hf_dataset, eval_transform)

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=2)

        logging.info("Dataloaders created successfully.")
        return train_loader, val_loader

    except Exception as e:
        logging.error(f"Failed to load dataset '{train_dataset_name}': {e}")
        logging.error("Please ensure you have accepted the terms on the Hugging Face dataset page and have a valid HF_TOKEN.")
        raise
