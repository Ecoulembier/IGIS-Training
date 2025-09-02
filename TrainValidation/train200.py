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
        mask[mask > 0] = 1.0  # Threshold: Set all values > 0 to 1
        mask = torch.from_numpy(mask)
        mask = torch.unsqueeze(mask, dim=0)

        return image, mask

# Set hyperparameters
IMG_SIZE = (350, 350)
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
EPOCHS = 200
CHECKPOINT_EPOCHS = 40

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a segmentation model.")
    parser.add_argument("dataset_dir", help="Path to the directory containing training data (with img/ and masks_machine/ subdirectories)")
    args = parser.parse_args()

    # Create checkpoint directory
    checkpoint_dir = "trained_model_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Construct paths from dataset directory
    train_img_dir = os.path.join(args.dataset_dir, 'img')
    train_mask_dir = os.path.join(args.dataset_dir, 'masks_machine')

    # Load data
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
    ])

    dataset = SegmentationDataset(train_img_dir, train_mask_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize pre-trained ResNet50 backbone
    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    backbone = models._utils.IntermediateLayerGetter(backbone, return_layers={'layer4': 'out'})

    # Define a classifier for DeepLabV3
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

    # Initialize DeepLabV3 model
    in_channels = 2048  # Output channels of ResNet50 layer4
    num_classes = 1     # For binary segmentation
    classifier = DeepLabClassifier(in_channels, num_classes)
    model = deeplabv3.DeepLabV3(backbone, classifier)

    # Set device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Metrics
    iou = JaccardIndex(task="binary").to(device)
    precision = Precision(task="binary").to(device)
    recall = Recall(task="binary").to(device)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        total_iou = []
        total_precision = []
        total_recall = []

        for batch_idx, (images, masks) in enumerate(data_loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)['out']
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Calculate metrics for this batch
            outputs = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
            outputs = (outputs > 0.5).float()  # Threshold to get binary predictions

            iou_val = iou(outputs, masks)
            precision_val = precision(outputs, masks)
            recall_val = recall(outputs, masks)

            total_iou.append(iou_val.item())
            total_precision.append(precision_val.item())
            total_recall.append(recall_val.item())

        # Calculate average metrics for the epoch
        avg_iou = np.mean(total_iou)
        avg_precision = np.mean(total_precision)
        avg_recall = np.mean(total_recall)
        
        current_lr = optimizer.param_groups[0]['lr']

        print(f'Epoch {epoch+1}, LR: {current_lr}, Loss: {epoch_loss / len(data_loader)}, IoU: {avg_iou}, Precision: {avg_precision}, Recall: {avg_recall}')

        # Save checkpoint
        if (epoch + 1) % CHECKPOINT_EPOCHS == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved at {checkpoint_path}')

if __name__ == "__main__":
    main()
