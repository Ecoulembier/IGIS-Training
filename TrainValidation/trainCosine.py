import os
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.segmentation import deeplabv3
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics import JaccardIndex, Precision, Recall
from torch.optim.lr_scheduler import LambdaLR   ### NEW
import numpy as np
from PIL import Image
import sys
import argparse
import math   ### NEW

# Define custom dataset class
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

        # Convert image to tensor
        image = transforms.ToTensor()(image)

        # Ensure masks are strictly 0 or 1
        mask = np.array(mask, dtype=np.float32)
        mask[mask > 0] = 1.0
        mask = torch.from_numpy(mask)
        mask = torch.unsqueeze(mask, dim=0)

        return image, mask

# Set hyperparameters
IMG_SIZE = (350, 350)
BASE_LR = 1e-1   ### CHANGED: renamed for clarity
BATCH_SIZE = 16
EPOCHS = 200
CHECKPOINT_EPOCHS = 40
WARMUP_EPOCHS = 20   ### NEW

def main():
    parser = argparse.ArgumentParser(description="Train a segmentation model.")
    parser.add_argument("dataset_dir", help="Path to dataset (with img/ and masks_machine/ subdirs)")
    args = parser.parse_args()

    checkpoint_dir = "trained_model_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_img_dir = os.path.join(args.dataset_dir, 'img')
    train_mask_dir = os.path.join(args.dataset_dir, 'masks_machine')

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
    ])

    dataset = SegmentationDataset(train_img_dir, train_mask_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Backbone
    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    backbone = models._utils.IntermediateLayerGetter(backbone, return_layers={'layer4': 'out'})

    # DeepLab classifier
    class DeepLabClassifier(nn.Module):
        def __init__(self, in_channels, num_classes):
            super().__init__()
            self.aspp = nn.Sequential(
                nn.Conv2d(in_channels, 256, kernel_size=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 256, kernel_size=3, padding=6, dilation=6),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 256, kernel_size=3, padding=12, dilation=12),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 256, kernel_size=3, padding=18, dilation=18),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 256, kernel_size=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
            )
            self.project = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, num_classes, kernel_size=1)
            )

        def forward(self, x):
            x = self.aspp(x)
            x = self.project(x)
            x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)(x)
            return x

    in_channels = 2048
    num_classes = 1
    classifier = DeepLabClassifier(in_channels, num_classes)
    model = deeplabv3.DeepLabV3(backbone, classifier)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Loss + Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(   ### CHANGED (Adam ? SGD for warmup+cosine)
        model.parameters(), lr=BASE_LR, momentum=0.9, weight_decay=1e-4
    )

    # Warmup + Cosine LR Scheduler
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return float(epoch) / float(max(1, WARMUP_EPOCHS))
        progress = float(epoch - WARMUP_EPOCHS) / float(max(1, EPOCHS - WARMUP_EPOCHS))
        return 0.5 * (1. + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # Metrics
    iou = JaccardIndex(task="binary").to(device)
    precision = Precision(task="binary").to(device)
    recall = Recall(task="binary").to(device)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        total_iou, total_precision, total_recall = [], [], []

        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Metrics
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > 0.5).float()
            total_iou.append(iou(outputs, masks).item())
            total_precision.append(precision(outputs, masks).item())
            total_recall.append(recall(outputs, masks).item())

        scheduler.step()   ### NEW: update LR

        avg_iou = np.mean(total_iou)
        avg_precision = np.mean(total_precision)
        avg_recall = np.mean(total_recall)

        print(f'Epoch {epoch+1}, '
              f'Loss: {epoch_loss / len(data_loader):.4f}, '
              f'IoU: {avg_iou:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, '
              f'LR: {scheduler.get_last_lr()[0]:.6f}')   ### NEW: log LR

        if (epoch + 1) % CHECKPOINT_EPOCHS == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
              'epoch': epoch + 1,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            print(f'Checkpoint saved at {checkpoint_path}')


if __name__ == "__main__":
    main()
