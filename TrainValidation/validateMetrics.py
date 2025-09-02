import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models.segmentation import deeplabv3
from torch.utils.data import Dataset, DataLoader
from torchmetrics import JaccardIndex, Precision, Recall
import numpy as np
from PIL import Image
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

# Confusion matrix counts
def confusion_matrix_counts(preds: torch.Tensor, targets: torch.Tensor):
    preds = preds.view(-1).long()
    targets = targets.view(-1).long()

    TP = ((preds == 1) & (targets == 1)).sum().item()
    TN = ((preds == 0) & (targets == 0)).sum().item()
    FP = ((preds == 1) & (targets == 0)).sum().item()
    FN = ((preds == 0) & (targets == 1)).sum().item()

    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN}

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
        mask[mask > 0] = 1.0
        mask = torch.from_numpy(mask)
        mask = torch.unsqueeze(mask, dim=0)
        return image, mask

# Set hyperparameters
IMG_SIZE = (350, 350)
BATCH_SIZE = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE)
])

def main():
    parser = argparse.ArgumentParser(description="Validate a segmentation model on a test set.")
    parser.add_argument("testset_dir", help="Path to the test set dir (with img/ and masks_machine/)")
    parser.add_argument("checkpoint_path", help="Path to trained model checkpoint (.pth)")
    args = parser.parse_args()

    testset_dir = args.testset_dir
    CHECKPOINT_PATH = args.checkpoint_path
    TEST_IMG_DIR = os.path.join(testset_dir, 'img')
    TEST_MASK_DIR = os.path.join(testset_dir, 'masks_machine')

    test_dataset = SegmentationDataset(TEST_IMG_DIR, TEST_MASK_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
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

    # --- Load checkpoint ---
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    # Some checkpoints save only weights, some save full dict with metadata.
    if "model_state_dict" in checkpoint:
        print("Loading model_state_dict from checkpoint...")
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Loading raw state_dict from checkpoint...")
        model.load_state_dict(checkpoint)

    # Metrics
    iou = JaccardIndex(task="binary").to(DEVICE)
    precision = Precision(task="binary").to(DEVICE)
    recall = Recall(task="binary").to(DEVICE)

    total_iou, total_precision, total_recall = [], [], []
    all_confusion = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    output_dir = "image_predictions"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "prediction_areas.txt"), "w") as area_file:
        area_file.write("Image Name,Predicted Area\n")

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)['out']
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > 0.5).float()

            iou_val = iou(outputs, masks)
            precision_val = precision(outputs, masks)
            recall_val = recall(outputs, masks)

            total_iou.append(iou_val.item())
            total_precision.append(precision_val.item())
            total_recall.append(recall_val.item())

            # Confusion matrix accumulation
            cm = confusion_matrix_counts(outputs, masks)
            for k in all_confusion:
                all_confusion[k] += cm[k]

            # Image names
            batch_image_names = [test_dataset.images[batch_idx * BATCH_SIZE + i] for i in range(images.shape[0])]

            # Prediction areas + visualization
            for img, mask, pred, name in zip(images, masks, outputs, batch_image_names):
                pred_area = pred.sum().item()
                with open(os.path.join(output_dir, "prediction_areas.txt"), "a") as area_file:
                    area_file.write(f"{name},{pred_area}\n")

                overlay = draw_segmentation_masks(
                    (img * 255).type(torch.uint8).cpu(),
                    masks=torch.cat([(mask > 0.5).cpu(), (pred > 0.5).cpu()], dim=0),
                    alpha=0.5,
                    colors=["green", "red"]
                )
                plt.imshow(overlay.permute(1,2,0).cpu().numpy())
                plt.axis('off')
                plt.savefig(os.path.join(output_dir, f"{name}_prediction.png"), bbox_inches='tight')
                plt.close()

    # Average + std metrics
    avg_iou, std_iou = np.mean(total_iou), np.std(total_iou)
    avg_precision, std_precision = np.mean(total_precision), np.std(total_precision)
    avg_recall, std_recall = np.mean(total_recall), np.std(total_recall)

    print(f'IoU: {avg_iou:.4f} ± {std_iou:.4f}')
    print(f'Precision: {avg_precision:.4f} ± {std_precision:.4f}')
    print(f'Recall: {avg_recall:.4f} ± {std_recall:.4f}')
    print(f'Confusion Matrix: {all_confusion}')

    # Derived metrics
    TP, TN, FP, FN = all_confusion["TP"], all_confusion["TN"], all_confusion["FP"], all_confusion["FN"]
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    f1_score = 2 * TP / (2 * TP + FP + FN + 1e-8)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1-score: {f1_score:.4f}')

    # Save metrics to CSV
    output_file = os.path.join(output_dir, 'validation_metrics.csv')
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Metric', 'Mean', 'Std', 'Global'])

        writer.writerow(['IoU', avg_iou, std_iou, ''])
        writer.writerow(['Precision', avg_precision, std_precision, ''])
        writer.writerow(['Recall', avg_recall, std_recall, ''])
        writer.writerow(['Accuracy', '', '', accuracy])
        writer.writerow(['F1-score', '', '', f1_score])
        writer.writerow(['TP', '', '', TP])
        writer.writerow(['TN', '', '', TN])
        writer.writerow(['FP', '', '', FP])
        writer.writerow(['FN', '', '', FN])

        writer.writerow(['IoU values'] + total_iou)
        writer.writerow(['Precision values'] + total_precision)
        writer.writerow(['Recall values'] + total_recall)

    print(f'Validation metrics saved to {output_file}')

if __name__ == "__main__":
    main()
