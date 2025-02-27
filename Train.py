import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from loss import CombinedQuantumLoss
from wsi_training import WSIDataset
from model import WSIResNet
from hparam_optimization import HyperparameterTuner

class MetricsTracker:
    def __init__(self, save_dir='./plots'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.train_losses, self.val_losses = [], []
        self.train_accs, self.val_accs = [], []
        self.train_class_accs, self.val_class_accs = [], []

    def update(self, train_loss, train_acc, train_class_acc, val_loss, val_acc, val_class_acc):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.train_class_accs.append(train_class_acc)
        self.val_class_accs.append(val_class_acc)

def compute_class_accuracies(all_preds, all_labels, num_classes=4):
    cm = confusion_matrix(all_labels.cpu(), all_preds.cpu(), labels=range(num_classes))
    return cm.diagonal() / cm.sum(axis=1) * 100

def train_one_epoch(model, train_loader, criterion, optimizer, device, scheduler, scaler):
    model.train()
    running_loss, running_correct, running_total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for batch in train_loader:
        tiles, labels = batch['tiles'].to(device), batch['label'].to(device)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            features, outputs = model(tiles)
            loss = criterion(features, outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        running_correct += predicted.eq(labels).sum().item()
        running_total += labels.size(0)
        all_preds.extend(predicted.cpu())
        all_labels.extend(labels.cpu())

    return running_loss / len(train_loader), 100. * running_correct / running_total, compute_class_accuracies(torch.tensor(all_preds), torch.tensor(all_labels))

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss, running_correct, running_total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        for batch in val_loader:
            tiles, labels = batch['tiles'].to(device), batch['label'].to(device)
            features, outputs = model(tiles)
            loss = criterion(features, outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            running_correct += predicted.eq(labels).sum().item()
            running_total += labels.size(0)
            all_preds.extend(predicted.cpu())
            all_labels.extend(labels.cpu())

    return running_loss / len(val_loader), 100. * running_correct / running_total, compute_class_accuracies(torch.tensor(all_preds), torch.tensor(all_labels))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    csv_path, wsi_dir, tiles_dir = "D:/Meghdad/Path-Lung-BMIRDS/MetaData_Release_1.0.csv", "D:/Meghdad/Path-Lung-BMIRDS/DHMC_wsi", "D:/Meghdad/Path-Lung-BMIRDS/tiles"

    train_dataset = WSIDataset(csv_path, wsi_dir, tiles_dir, batch_size=16, is_training=True)
    val_dataset = WSIDataset(csv_path, wsi_dir, tiles_dir, batch_size=16, is_training=False)
    train_indices, val_indices = train_test_split(range(len(train_dataset)), test_size=0.2, stratify=[train_dataset.wsi_batches[i]['class'] for i in range(len(train_dataset))], random_state=42)

    train_loader = DataLoader(train_dataset, batch_size=8, sampler=torch.utils.data.SubsetRandomSampler(train_indices), num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, sampler=torch.utils.data.SubsetRandomSampler(val_indices), num_workers=0, pin_memory=True)

    model = WSIResNet(backbone='resnet50').to(device)
    criterion = CombinedQuantumLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = OneCycleLR(optimizer, max_lr=3e-4, steps_per_epoch=len(train_loader), epochs=50)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(50):
        train_loss, train_acc, train_class_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scheduler, scaler)
        val_loss, val_acc, val_class_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.2f}%, Val Loss {val_loss:.4f}, Val Acc {val_acc:.2f}%")

if __name__ == "__main__":
    main()
