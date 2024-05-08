import torch
import torch.nn as nn
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from models import *
from training_statistics import TrainingDataCollector, TrainingDataPlotter


def train(model, optimizer, scheduler, criterion, train_dataloader, test_dataloader, n_epoch=10, device='cpu'):
    history_lr = []
    tdc = TrainingDataCollector()
    for epoch in range(1, n_epoch+1):
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []

        model.train()
        for x, y in tqdm(train_dataloader, desc=f'Training epoch {epoch}'):
            x = x.to(device)

            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            history_lr.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

            optimizer.zero_grad()
            train_loss.append(loss.item())
            train_acc.append((torch.argmax(y_pred.cpu(), dim=-1) == y).numpy().tolist())

        model.eval()
        for x, y in tqdm(test_dataloader, desc=f'Testing epoch {epoch}'):
            x = x.to(device)

            with torch.no_grad():
                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_loss.append(loss.item())
                val_acc.append((torch.argmax(y_pred.cpu(), dim=-1) == y).numpy().tolist())

        tl, ta = np.mean(train_loss), np.mean(train_acc)
        vl, va = np.mean(val_loss), np.mean(val_acc)
        tdc.update_statistics(tl, ta, vl, va)
        print(f'{epoch=}')
        print(f'TRAIN: loss={tl:.6f}, acc={ta:.6f}')
        print(f' TEST: loss={vl:.6f}, acc={va:.6f}')

    return tdc, history_lr

if __name__ == "__main__":
    classes = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_mynet_model()
    model = model.to(device)
    path_to_dataset = 'path/to/augmented/dataset'
    EPOCHS = 30
    train_transform = v2.Compose([
                                  v2.Grayscale(1),
                                  v2.RandomHorizontalFlip(p=0.2),
                                  v2.ToDtype(torch.float32, scale=True),
                                  v2.Normalize(mean=(0.4809,), std=(0.2066,)
                                 ])

    val_transform = v2.Compose([
                                v2.Grayscale(1),
                                v2.ToDtype(torch.float32, scale=True),
                                v2.Normalize(mean=(0.4809,), std=(0.2066,)
                               ])

    train_ds = ImageFolder(path_to_dataset, transform=train_transform)
    val_ds = ImageFolder(path_to_dataset, transform=val_transform)

    train_dataloader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(val_ds, batch_size=8, shuffle=True, drop_last=True)

    optimizer = torch.optim.NAdam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.003, steps_per_epoch=len(train_dataloader), EPOCHS)
    criterion = nn.CrossEntropyLoss()

    tdc, _ = train(model, optimizer, scheduler, criterion, train_dataloader, test_dataloader, EPOCHS, device)

    tdp = TrainingDataPlotter(tdc)
    tdp.plot_statistics()





