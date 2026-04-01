# -*- coding: utf-8 -*-


from google.colab import drive
drive.mount('/content/drive')

!unzip -q "/content/drive/MyDrive/AI_dataset/archive.zip" -d "/content/dataset"

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import v2

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score



# 1. БАЗОВЫЕ НАСТРОЙКИ


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


DATA_DIR = Path("/content/dataset/images/Images")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3
BEST_MODEL_PATH = "/content/best_mobilenetv2_custom_scratch.pth"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)
print("Device:", DEVICE)

assert DATA_DIR.exists(), f"Папка {DATA_DIR} не найдена"



# 2. ПОДГОТОВКА ДАННЫХ


classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}
num_classes = len(classes)

print("Количество классов:", num_classes)

image_paths = []
labels = []

for cls_name in classes:
    class_dir = DATA_DIR / cls_name
    for img_path in class_dir.glob("*"):
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            image_paths.append(str(img_path))
            labels.append(class_to_idx[cls_name])

image_paths = np.array(image_paths)
labels = np.array(labels)

print("Количество изображений:", len(image_paths))

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.30,
    stratify=labels,
    random_state=SEED
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.50,
    stratify=temp_labels,
    random_state=SEED
)

print("Train:", len(train_paths))
print("Val:", len(val_paths))
print("Test:", len(test_paths))



# 3. DATASET И ТРАНСФОРМАЦИИ


class StanfordDogsDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


train_transform = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=15),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])

eval_transform = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])

train_dataset = StanfordDogsDataset(train_paths, train_labels, transform=train_transform)
val_dataset = StanfordDogsDataset(val_paths, val_labels, transform=eval_transform)
test_dataset = StanfordDogsDataset(test_paths, test_labels, transform=eval_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)



# 4. РУЧНАЯ РЕАЛИЗАЦИЯ MOBILE NET V2


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):

    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        assert stride in [1, 2]

        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

        layers = []

        # 1. Expansion layer
        if expand_ratio != 1:
            layers.append(
                ConvBNReLU(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    stride=1
                )
            )

        # 2. Depthwise convolution
        layers.append(
            ConvBNReLU(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=stride,
                groups=hidden_dim
            )
        )

        # 3. Projection layer
        layers.append(
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
        )
        layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        return self.block(x)


class MobileNetV2Custom(nn.Module):

    def __init__(self, num_classes=120, width_mult=1.0, dropout=0.2):
        super().__init__()

        input_channel = 32
        last_channel = 1280

        inverted_residual_setting = [
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))

        features = []

        # Начальный слой
        features.append(
            ConvBNReLU(
                in_channels=3,
                out_channels=input_channel,
                kernel_size=3,
                stride=2
            )
        )

        # Основные inverted residual блоки
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)

            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidual(
                        in_channels=input_channel,
                        out_channels=output_channel,
                        stride=stride,
                        expand_ratio=t
                    )
                )
                input_channel = output_channel

        # Финальный 1x1 conv
        features.append(
            ConvBNReLU(
                in_channels=input_channel,
                out_channels=self.last_channel,
                kernel_size=1,
                stride=1
            )
        )

        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


model = MobileNetV2Custom(num_classes=num_classes, width_mult=1.0, dropout=0.2).to(DEVICE)



# 5. LOSS И OPTIMIZER


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)



# 6. ФУНКЦИИ ОБУЧЕНИЯ И ОЦЕНКИ


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc, np.array(all_targets), np.array(all_preds)



# 7. ОБУЧЕНИЕ

best_val_loss = float("inf")

history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] | "
        f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
        f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
    )



# 8. ЗАГРУЗКА ЛУЧШЕЙ МОДЕЛИ


model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))



# 9. ОЦЕНКА НА TEST: PRECISION / RECALL / F1-SCORE


test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, DEVICE)

precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

print("\nРезультаты на тестовой выборке:")
print(f"Test Loss:  {test_loss:.4f}")
print(f"Test Acc:   {test_acc:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall:     {recall:.4f}")
print(f"F1-score:   {f1:.4f}")



# 10. ИТОГОВАЯ ТАБЛИЦА


results_df = pd.DataFrame([
    {
        "Model": "MobileNetV2 custom from scratch + Adam",
        "Test Loss": test_loss,
        "Test Accuracy": test_acc,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }
])

print("\nИтоговая таблица:")
print(results_df)

test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)

# Основные метрики классификации
precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

print("Результаты на тестовой выборке:")
print(f"Test Loss:  {test_loss:.4f}")
print(f"Test Acc:   {test_acc:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall:     {recall:.4f}")
print(f"F1-score:   {f1:.4f}")
