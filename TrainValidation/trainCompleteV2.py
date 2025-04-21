import os
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.segmentation import deeplabv3
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics import JaccardIndex, Precision, Recall
import numpy as np
from PIL import Image
import sys
import argparse

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        image = transforms.ToTensor()(image)
        mask = np.array(mask, dtype=np.float32)
        mask[mask > 0] = 1.0
        mask = torch.from_numpy(mask)
        mask = torch.unsqueeze(mask, dim=0)

        return image, mask

def main():
    parser = argparse.ArgumentParser(description="Train a segmentation model.")
    parser.add_argument("dataset_dir", help="Path to training data directory")
    parser.add_argument("--checkpoint", help="Path to model checkpoint")
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    # Constants
    IMG_SIZE = (350, 350)
    CHECKPOINT_EPOCHS = 5

    # Setup
    checkpoint_dir = "trained_model_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loading
    transform = transforms.Compose([transforms.Resize(IMG_SIZE)])
    train_img_dir = os.path.join(args.dataset_dir, 'img')
    train_mask_dir = os.path.join(args.dataset_dir, 'masks_machine')
    dataset = SegmentationDataset(train_img_dir, train_mask_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model initialization
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model = deeplabv3.DeepLabV3(
            backbone=models._utils.IntermediateLayerGetter(
                models.resnet50(weights=None),
                return_layers={'layer4': 'out'}
            ),
            classifier=nn.Sequential()
        ).to(device)
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Resuming training from epoch {start_epoch}")
        else:
            model.load_state_dict(checkpoint)
            start_epoch = 0
            print("Loaded model weights (legacy checkpoint format)")
    else:
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        backbone = models._utils.IntermediateLayerGetter(backbone, return_layers={'layer4': 'out'})
        model = deeplabv3.DeepLabV3(backbone, nn.Sequential()).to(device)
        start_epoch = 0

    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.checkpoint and isinstance(checkpoint, dict):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Metrics
    iou = JaccardIndex(task="binary").to(device)
    precision = Precision(task="binary").to(device)
    recall = Recall(task="binary").to(device)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        total_iou = []
        total_precision = []
        total_recall = []

        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > 0.5).float()

            total_iou.append(iou(outputs, masks).item())
            total_precision.append(precision(outputs, masks).item())
            total_recall.append(recall(outputs, masks).item())

        # Epoch statistics
        avg_iou = np.mean(total_iou)
        avg_precision = np.mean(total_precision)
        avg_recall = np.mean(total_recall)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss/len(data_loader):.4f}, "
              f"IoU: {avg_iou:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}")

        # Checkpoint saving
        if (epoch + 1) % CHECKPOINT_EPOCHS == 0 or epoch + 1 == args.epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss / len(data_loader),
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    main()
