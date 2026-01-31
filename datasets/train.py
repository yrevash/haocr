"""
TRAINING SCRIPT WITH WANDB LOGGING
Logs: Loss, Accuracy, mAP, Learning Rate
For Assignment: Shows proper experiment tracking
"""

import os
import json
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms, models, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import random
import wandb

# ============================================
# INITIALIZE WANDB
# ============================================
wandb.init(
    project="human-animal-detection",
    name="two-model-detection-run",
    config={
        "animal_limit": 2500,
        "detection_epochs": 10,
        "classifier_epochs": 15,
        "batch_size_detection": 4,
        "batch_size_classifier": 64,
        "detection_lr": 0.005,
        "classifier_lr": 0.0003,
        "nms_threshold": 0.3,
        "confidence_threshold": 0.6,
    }
)

config = wandb.config

# ============================================
# CONFIGURATION
# ============================================
ANIMAL_LIMIT = config.animal_limit
DETECTION_EPOCHS = config.detection_epochs
CLASSIFIER_EPOCHS = config.classifier_epochs
BATCH_SIZE_DETECTION = config.batch_size_detection
BATCH_SIZE_CLASSIFIER = config.batch_size_classifier
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")
print(f"Logging to WandB: {wandb.run.name}")

# ============================================
# STEP 1: DOWNLOAD FROM ROBOFLOW
# ============================================
print("\n" + "="*60)
print("STEP 1: Downloading Dataset")
print("="*60)

from roboflow import Roboflow

rf = Roboflow(api_key="")
project = rf.workspace("yrevash-7tnv7").project("humans-and-animals-detection-vv9pj")
version = project.version(2)
dataset = version.download("createml")

DATASET_ROOT = dataset.location
print(f" Dataset downloaded to: {DATASET_ROOT}")
# DATASET_ROOT = "Humans-and-Animals-Detection-2"
# print(f"✅ Dataset downloaded to: {DATASET_ROOT}")

# ============================================
# STEP 2: BALANCE DATASET
# ============================================
print("\n" + "="*60)
print("STEP 2: Balancing Dataset")
print("="*60)

def parse_and_balance_createml(json_path, img_dir, max_animals=ANIMAL_LIMIT):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    human_anns = []
    animal_anns = []
    
    for item in data:
        img_name = item['image']
        anns = item.get('annotations', [])
        
        if not anns:
            continue
        
        boxes = []
        labels = []
        has_human = False
        
        for ann in anns:
            label = ann['label']
            coords = ann['coordinates']
            
            x_center = coords['x']
            y_center = coords['y']
            width = coords['width']
            height = coords['height']
            
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            boxes.append([x1, y1, x2, y2])
            label_idx = 1 if label == 'animal' else 2
            labels.append(label_idx)
            
            if label == 'human':
                has_human = True
        
        ann_dict = {
            'image': img_name,
            'boxes': boxes,
            'labels': labels
        }
        
        if has_human:
            human_anns.append(ann_dict)
        else:
            animal_anns.append(ann_dict)
    
    print(f"  Original - Humans: {len(human_anns)}, Animals: {len(animal_anns)}")
    
    if len(animal_anns) > max_animals:
        random.shuffle(animal_anns)
        animal_anns = animal_anns[:max_animals]
    
    balanced_anns = human_anns + animal_anns
    random.shuffle(balanced_anns)
    
    # Log dataset stats to WandB
    wandb.log({
        "dataset/human_samples": len(human_anns),
        "dataset/animal_samples": len(animal_anns),
        "dataset/total_samples": len(balanced_anns)
    })
    
    return balanced_anns

train_anns = parse_and_balance_createml(
    os.path.join(DATASET_ROOT, 'train', '_annotations.createml.json'),
    os.path.join(DATASET_ROOT, 'train')
)

val_anns = parse_and_balance_createml(
    os.path.join(DATASET_ROOT, 'valid', '_annotations.createml.json'),
    os.path.join(DATASET_ROOT, 'valid'),
    max_animals=500
)

print(f"✅ Training samples: {len(train_anns)}")
print(f"✅ Validation samples: {len(val_anns)}")

# ============================================
# STEP 3: DETECTION DATASET
# ============================================

class DetectionDataset(Dataset):
    def __init__(self, root_dir, annotations, transforms=None):
        self.root = root_dir
        self.annotations = annotations
        self.transforms = transforms
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = os.path.join(self.root, ann['image'])
        
        img = Image.open(img_path).convert('RGB')
        boxes = torch.as_tensor(ann['boxes'], dtype=torch.float32)
        labels = torch.as_tensor(ann['labels'], dtype=torch.int64)
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        
        if self.transforms:
            img = self.transforms(img)
        else:
            img = transforms.ToTensor()(img)
        
        return img, target

train_dataset = DetectionDataset(
    os.path.join(DATASET_ROOT, 'train'),
    train_anns
)

val_dataset = DetectionDataset(
    os.path.join(DATASET_ROOT, 'valid'),
    val_anns
)

def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE_DETECTION,
    shuffle=True,
    num_workers=8,
    collate_fn=collate_fn,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE_DETECTION,
    shuffle=False,
    num_workers=8,
    collate_fn=collate_fn,
    pin_memory=True
)

# ============================================
# STEP 4: TRAIN FASTER R-CNN WITH LOGGING
# ============================================
print("\n" + "="*60)
print("STEP 4: Training Faster R-CNN (Detection Model)")
print("="*60)

detector = fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 3
in_features = detector.roi_heads.box_predictor.cls_score.in_features
detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
detector = detector.to(DEVICE)

# Watch model with WandB
wandb.watch(detector, log="all", log_freq=100)

params = [p for p in detector.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=config.detection_lr,
    momentum=0.9,
    weight_decay=0.0005
)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

for epoch in range(DETECTION_EPOCHS):
    detector.train()
    epoch_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{DETECTION_EPOCHS}")
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        loss_dict = detector(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        epoch_loss += losses.item()
        pbar.set_postfix({'loss': f"{losses.item():.4f}"})
        
        # Log batch-level metrics to WandB
        if batch_idx % 10 == 0:
            wandb.log({
                "detection/batch_loss": losses.item(),
                "detection/loss_classifier": loss_dict['loss_classifier'].item(),
                "detection/loss_box_reg": loss_dict['loss_box_reg'].item(),
                "detection/loss_objectness": loss_dict['loss_objectness'].item(),
                "detection/loss_rpn_box_reg": loss_dict['loss_rpn_box_reg'].item(),
            })
    
    lr_scheduler.step()
    avg_loss = epoch_loss / len(train_loader)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Log epoch-level metrics
    wandb.log({
        "detection/epoch": epoch + 1,
        "detection/epoch_loss": avg_loss,
        "detection/learning_rate": current_lr
    })
    
    print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
    
    if (epoch + 1) % 3 == 0:
        torch.save(detector.state_dict(), f'detector_epoch{epoch+1}.pth')
        # Save to WandB
        wandb.save(f'detector_epoch{epoch+1}.pth')

torch.save(detector.state_dict(), 'detector_final.pth')
wandb.save('detector_final.pth')
print("✅ Faster R-CNN training complete!")

# ============================================
# STEP 5: CREATE CLASSIFICATION DATASET
# ============================================
print("\n" + "="*60)
print("STEP 5: Extracting Crops for Classification")
print("="*60)

CROP_DIR = "classification_dataset"
os.makedirs(f"{CROP_DIR}/train/human", exist_ok=True)
os.makedirs(f"{CROP_DIR}/train/animal", exist_ok=True)
os.makedirs(f"{CROP_DIR}/valid/human", exist_ok=True)
os.makedirs(f"{CROP_DIR}/valid/animal", exist_ok=True)

def extract_crops_balanced(annotations, img_dir, output_dir, max_per_class=3000):
    human_count = 0
    animal_count = 0
    
    for ann in tqdm(annotations, desc="Extracting crops"):
        img_path = os.path.join(img_dir, ann['image'])
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        for i, (box, label) in enumerate(zip(ann['boxes'], ann['labels'])):
            label_name = 'animal' if label == 1 else 'human'
            
            if label_name == 'animal' and animal_count >= max_per_class:
                continue
            if label_name == 'human' and human_count >= max_per_class:
                continue
            
            x1, y1, x2, y2 = map(int, box)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            crop = img[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            
            crop_name = f"{os.path.splitext(ann['image'])[0]}_{i}.jpg"
            save_path = os.path.join(output_dir, label_name, crop_name)
            
            cv2.imwrite(save_path, crop)
            
            if label_name == 'animal':
                animal_count += 1
            else:
                human_count += 1
    
    return animal_count, human_count

train_animal, train_human = extract_crops_balanced(
    train_anns, 
    os.path.join(DATASET_ROOT, 'train'), 
    f"{CROP_DIR}/train"
)

val_animal, val_human = extract_crops_balanced(
    val_anns, 
    os.path.join(DATASET_ROOT, 'valid'), 
    f"{CROP_DIR}/valid", 
    max_per_class=1000
)

# Log crop extraction stats
wandb.log({
    "classification/train_animal_crops": train_animal,
    "classification/train_human_crops": train_human,
    "classification/val_animal_crops": val_animal,
    "classification/val_human_crops": val_human
})

print("✅ Classification dataset created!")

# ============================================
# STEP 6: TRAIN CLASSIFIER WITH LOGGING
# ============================================
print("\n" + "="*60)
print("STEP 6: Training EfficientNet Classifier")
print("="*60)

train_transforms = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.RandomResizedCrop(300),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset_clf = datasets.ImageFolder(f"{CROP_DIR}/train", transform=train_transforms)
val_dataset_clf = datasets.ImageFolder(f"{CROP_DIR}/valid", transform=val_transforms)

print(f"Classes: {train_dataset_clf.class_to_idx}")

train_loader_clf = DataLoader(
    train_dataset_clf,
    batch_size=BATCH_SIZE_CLASSIFIER,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)

val_loader_clf = DataLoader(
    val_dataset_clf,
    batch_size=BATCH_SIZE_CLASSIFIER,
    shuffle=False,
    num_workers=8,
    pin_memory=True
)

classifier = models.efficientnet_b3(weights='DEFAULT')
classifier.classifier[1] = nn.Linear(classifier.classifier[1].in_features, 2)
classifier = classifier.to(DEVICE)

# Watch classifier with WandB
wandb.watch(classifier, log="all", log_freq=50)

# Class weights
animal_count = len([f for f in os.listdir(f"{CROP_DIR}/train/animal")])
human_count = len([f for f in os.listdir(f"{CROP_DIR}/train/human")])
total = animal_count + human_count

weight_animal = total / (2 * animal_count)
weight_human = total / (2 * human_count)
class_weights = torch.tensor([weight_animal, weight_human]).to(DEVICE)

wandb.log({
    "classification/class_weight_animal": weight_animal,
    "classification/class_weight_human": weight_human
})

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer_clf = torch.optim.AdamW(
    classifier.parameters(), 
    lr=config.classifier_lr, 
    weight_decay=0.01
)
scheduler_clf = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_clf, 
    T_max=CLASSIFIER_EPOCHS
)

best_acc = 0

for epoch in range(CLASSIFIER_EPOCHS):
    # Train
    classifier.train()
    train_loss = 0
    correct = 0
    total_samples = 0
    
    pbar = tqdm(train_loader_clf, desc=f"Epoch {epoch+1}/{CLASSIFIER_EPOCHS}")
    for batch_idx, (imgs, labels) in enumerate(pbar):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
        optimizer_clf.zero_grad()
        outputs = classifier(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_clf.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total_samples += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}", 
            'acc': f"{100.*correct/total_samples:.2f}%"
        })
        
        # Log batch metrics
        if batch_idx % 10 == 0:
            wandb.log({
                "classification/batch_loss": loss.item(),
                "classification/batch_accuracy": 100. * correct / total_samples
            })
    
    train_acc = 100. * correct / total_samples
    
    # Validation
    classifier.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0
    
    with torch.no_grad():
        for imgs, labels in val_loader_clf:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = classifier(imgs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_acc = 100. * val_correct / val_total
    avg_train_loss = train_loss / len(train_loader_clf)
    avg_val_loss = val_loss / len(val_loader_clf)
    current_lr = optimizer_clf.param_groups[0]['lr']
    
    # Log epoch metrics
    wandb.log({
        "classification/epoch": epoch + 1,
        "classification/train_loss": avg_train_loss,
        "classification/train_accuracy": train_acc,
        "classification/val_loss": avg_val_loss,
        "classification/val_accuracy": val_acc,
        "classification/learning_rate": current_lr
    })
    
    print(f"\nEpoch {epoch+1}:")
    print(f"  Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%")
    print(f"  Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(classifier.state_dict(), 'classifier_best.pth')
        wandb.save('classifier_best.pth')
        wandb.run.summary["best_val_accuracy"] = val_acc
        print(f" Saved best model (Val Acc: {val_acc:.2f}%)")
    
    scheduler_clf.step()

print(f"\nClassifier training complete! Best Val Accuracy: {best_acc:.2f}%")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)

wandb.run.summary["final_detection_loss"] = avg_loss
wandb.run.summary["final_classification_accuracy"] = best_acc

wandb.finish()