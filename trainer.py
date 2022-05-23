import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
from torch.utils.tensorboard import SummaryWriter

class Trainer():
    def __init__(self, model, device=torch.device('cpu')) -> None:
        self.model = model
        self.device = device

    def fit(self, dataset, batch_size, optimizer, criterion, epochs, scheduler=None, save_epoch=0, ckpt_dir='./ckpt/', val_epoch=0, val_dataset=None, save_best=False):
        writer = SummaryWriter()
        if save_epoch:
            os.makedirs(ckpt_dir, exist_ok=True)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        min_error = float('inf')
        for epoch in tqdm(epochs, desc="Train"):
            train_loss = 0
            for imgs, landmarks in train_loader:
                imgs, landmarks = imgs.to(self.device), landmarks.to(self.device)
                pred_landmarks = self.model(imgs)
                loss = criterion(pred_landmarks, landmarks)
                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if save_epoch and not (epoch + 1) % save_epoch:
                    self.save_model(path=os.path.join(ckpt_dir, f"checkpoint-{epoch + 1}.ckpt"))
            writer.add_scalar('Loss/train', train_loss / len(train_loader), epoch)

            if val_epoch and not (epoch + 1) % val_epoch and val_dataset:
                val_loss = self.validate(val_dataset)
                writer.add_scalar('Loss/validate', val_loss, epoch)
                if save_best and val_loss < min_error:
                    min_error = val_loss
                    self.save_model(path=os.path.join(ckpt_dir, "best.pt"))
                self.model.train()
            
            if scheduler is not None:
                scheduler.step()
        writer.close()


    def validate(self, dataset, batch_size, criterion):
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.model.eval()
        val_loss = 0
        for imgs, landmarks in val_loader:
            imgs, landmarks = imgs.to(self.device), landmarks.to(self.device)
            with torch.no_grad():
                pred_landmarks = self.model(imgs)
            loss = criterion(pred_landmarks, landmarks)
            val_loss += loss.item()
        return val_loss / len(val_loader)        

    def load_from_pretrained(self, path='./model.pt'):
        self.model.load_state_dict(torch.load(path))

    def save_model(self, path='./model.pt'):
        torch.save(self.model.state_dict(), path)

class NME(nn.Module):
    def __init__(self) -> None:
        super(NME, self).__init__()

    def forward(self, input, target):
        assert input.shape == target.shape
        dis = input - target
        dis = torch.sqrt(torch.sum(torch.pow(dis, 2), dim=-1))
        dis = torch.mean(dis)
        return dis / 384