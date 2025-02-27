import torch
import os
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from training_core import train_one_epoch, validate
from loss import CombinedQuantumLoss
from wsi_training import WSIDataset
from model import WSIResNet

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    csv_path, wsi_dir, tiles_dir = "", ""

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
