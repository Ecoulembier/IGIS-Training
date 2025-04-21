import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models.segmentation import deeplabv3
from torch.utils.data import Dataset, DataLoader
from torchmetrics import JaccardIndex, Precision, Recall
import numpy as np
from PIL import Image
import sys
import csv
import argparse
import matplotlib.pyplot as plt
from torchvision.utils import draw_segmentation_masks

# Custom IoU metric (as the built-in one had issues)
def iou_pytorch(inputs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6):
    inputs = inputs.contiguous().view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum().float()
    total = (inputs + targets).sum().float()
    union = total - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

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

        image = transforms.ToTensor()(image)

        # Ensure masks are strictly 0 or 1
        mask = np.array(mask, dtype=np.float32)
        mask[mask > 0] = 1.0  # Threshold: Set all values > 0 to 1
        mask = torch.from_numpy(mask)
        mask = torch.unsqueeze(mask, dim=0)
        return image, mask

# Set hyperparameters - adjust to match your training
IMG_SIZE = (350, 350)
BATCH_SIZE = 16 # Keep the same as training
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data - No shuffling for validation
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE)
])

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Validate a segmentation model on a test set.")
    parser.add_argument("testset_dir", help="Path to the directory containing test set (with img/ and masks_machine/ subdirectories)")
    parser.add_argument("checkpoint_path", help="Path to the trained model checkpoint file (.pth)")
    args = parser.parse_args()

    # Construct paths from the test set directory
    testset_dir = args.testset_dir
    CHECKPOINT_PATH = args.checkpoint_path
    TEST_IMG_DIR = os.path.join(testset_dir, 'img')
    TEST_MASK_DIR = os.path.join(testset_dir, 'masks_machine')

    test_dataset = SegmentationDataset(TEST_IMG_DIR, TEST_MASK_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize the model - must match your training model
    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    backbone = models._utils.IntermediateLayerGetter(backbone, return_layers={'layer4': 'out'})
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
    model.to(DEVICE)

    # Load the trained weights - now using command-line argument
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    # Metrics
    iou = JaccardIndex(task="binary").to(DEVICE) #Bring back IOU
    precision = Precision(task="binary").to(DEVICE)
    recall = Recall(task="binary").to(DEVICE)

    # Validation loop
    total_iou = []
    total_precision = []
    total_recall = []

    output_dir = "image_predictions"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "prediction_areas.txt"), "w") as area_file:
        area_file.write("Image Name,Predicted Area\n")

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)['out']
            outputs = torch.sigmoid(outputs) # Apply sigmoid
            outputs = (outputs > 0.5).float()  # Threshold

            iou_val = iou(outputs, masks)
            precision_val = precision(outputs, masks)
            recall_val = recall(outputs, masks)

            total_iou.append(iou_val.item())
            total_precision.append(precision_val.item())
            total_recall.append(recall_val.item())

            # Get image names for this batch
            batch_image_names = [test_dataset.images[batch_idx * BATCH_SIZE + i] for i in range(images.shape[0])]

            # Calculate and save the area of each prediction
            for img, mask, pred, name in zip(images, masks, outputs, batch_image_names):
                pred_area = pred.sum().item()
                with open(os.path.join(output_dir, "prediction_areas.txt"), "a") as area_file:
                    area_file.write(f"{name},{pred_area}\n")

                # Visualize and save the prediction
                overlay = draw_segmentation_masks(
                    (img * 255).type(torch.uint8).cpu(),
                    masks=torch.cat([(mask > 0.5).cpu(), (pred > 0.5).cpu()], dim=0),
                    alpha=0.5,
                    colors=["green", "red"]  # Green for GT, Red for prediction
                )
                plt.imshow(overlay.permute(1,2,0).cpu().numpy())
                plt.axis('off')
                plt.savefig(os.path.join(output_dir, f"{name}_prediction.png"), bbox_inches='tight')
                plt.close()

    # Calculate average metrics
    avg_iou = np.mean(total_iou)
    avg_precision = np.mean(total_precision)
    avg_recall = np.mean(total_recall)

    # Calculate standard deviations
    std_iou = np.std(total_iou)
    std_precision = np.std(total_precision)
    std_recall = np.std(total_recall)

    # Print metrics to console
    print(f'IoU: {avg_iou} ± {std_iou}')
    print(f'Precision: {avg_precision} ± {std_precision}')
    print(f'Recall: {avg_recall} ± {std_recall}')

    # Save metrics to a CSV file
    output_file = os.path.join(output_dir, 'validation_metrics.csv')
    with open(output_file, 'w', newline='') as csvfile:

        # Header
        writer = csv.writer(csvfile)
        writer.writerow(['Metric', 'Mean', 'Std'])

        # Write total metrics
        writer.writerow(['IoU', avg_iou, std_iou])
        writer.writerow(['Precision', avg_precision, std_precision])
        writer.writerow(['Recall', avg_recall, std_recall])

        # Write individual values
        writer.writerow(['IoU values'] + total_iou)
        writer.writerow(['Precision values'] + total_precision)
        writer.writerow(['Recall values'] + total_recall)

    print(f'Validation metrics saved to {output_file}')

if __name__ == "__main__":
    main()
