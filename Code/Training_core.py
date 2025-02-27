import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from sklearn.model_selection import train_test_split
from loss import CombinedQuantumLoss
from wsi_training import WSIDataset
from model import WSIResNet

def compute_class_accuracies(all_preds, all_labels, num_classes=4):
    cm = torch.zeros(num_classes, num_classes)
    for t, p in zip(all_labels, all_preds):
        cm[t.long(), p.long()] += 1
    return cm.diag() / cm.sum(1) * 100

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
