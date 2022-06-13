import torch
import torch.nn as nn
from tqdm import trange
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import os
from model import *
from pose import *
from torch.utils.tensorboard import SummaryWriter

class Trainer():
    def __init__(self, model, device=torch.device('cpu')) -> None:
        self.model = model.to(device)
        self.device = device

    def fit(self, dataset, batch_size, optimizer, criterion, num_epoch, save_epoch, scheduler=None, saved_dir='./ckpt/', val_epoch=0, val_dataset=None):
        writer = SummaryWriter(log_dir=saved_dir)
        if save_epoch:
            os.makedirs(saved_dir, exist_ok=True)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        min_error = float('inf')
        aux = AuxiliaryNet(input_channels=64).to(self.device)
        for epoch in trange(num_epoch, desc="Train", unit="epoch"):
            train_loss = 0
            self.model.train()
            aux.train()
            criterion = WingLoss(10.0, 2.0)
            for imgs, landmarks in tqdm(train_loader):
                imgs, landmarks = imgs.to(self.device), landmarks.to(self.device)
                pred_landmarks, aux_features = self.model(imgs)
                angle_preds = aux(aux_features)
                euler_angle_weights = get_euler_angle_weights(landmarks, angle_preds, self.device)
                loss = criterion(landmarks.to(self.device), pred_landmarks, euler_angle_weights)
                

                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (save_epoch and not (epoch + 1) % save_epoch) or epoch + 1 == num_epoch:
                    self.save_model(path=os.path.join(saved_dir, f"checkpoint-{epoch + 1}.ckpt"))
            writer.add_scalar('Loss/train', train_loss / len(train_loader), epoch + 1)

            if val_epoch and not (epoch + 1) % val_epoch and val_dataset is not None:
                val_loss = self.validate(val_dataset, batch_size, criterion)
                writer.add_scalar('Loss/validate', val_loss, epoch + 1)
                if val_loss < min_error:
                    min_error = val_loss
                    self.save_model(path=os.path.join(saved_dir, "best.pt"))
                self.model.train()
            
            if scheduler is not None:
                scheduler.step()
        writer.close()


    def validate(self, dataset, batch_size, criterion):
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.model.eval()
        val_loss = 0
        criterion = NME()
        for imgs, landmarks in val_loader:
            imgs, landmarks = imgs.to(self.device), landmarks.to(self.device)
            with torch.no_grad():
                pred_landmarks, _ = self.model(imgs)
            loss = criterion(pred_landmarks, landmarks)
            val_loss += loss.item()
        return val_loss / len(val_loader)        
    
    def predict(self, dataset, batch_size):
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        pred_id = []
        pred_lm = []
        self.model.eval()
        for ids, imgs in test_loader:
            _, _, H, W = imgs.shape
            pred_id += ids
            imgs = imgs.to(self.device)
            with torch.no_grad():
                pred_landmarks, _ = self.model(imgs)
            pred_landmarks = pred_landmarks.reshape(-1, 68, 2)
            pred_landmarks[:,:,0] *= W
            pred_landmarks[:,:,1] *= H
            pred_landmarks = pred_landmarks.type(torch.float32)
            pred_lm += pred_landmarks.view(pred_landmarks.size(0), -1).cpu().numpy().tolist()
        
        return pred_id, pred_lm

    def load_from_pretrained(self, path):
        self.model.load_state_dict(torch.load(path))

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    def get_euler_angle_weights(landmarks_batch, euler_angles_pre, device):
        TRACKED_POINTS = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]

        euler_angles_landmarks = []
        landmarks_batch = landmarks_batch.cpu().numpy()
        for index in TRACKED_POINTS:
            euler_angles_landmarks.append(landmarks_batch[:, 2 * index:2 * index + 2])
        euler_angles_landmarks = np.asarray(euler_angles_landmarks).transpose((1, 0, 2)).reshape((-1, 28))

        euler_angles_gt = []
        for j in range(euler_angles_landmarks.shape[0]):
            pitch, yaw, roll = calculate_pitch_yaw_roll(euler_angles_landmarks[j])
            euler_angles_gt.append((pitch, yaw, roll))
        euler_angles_gt = np.asarray(euler_angles_gt).reshape((-1, 3))

        euler_angles_gt = torch.Tensor(euler_angles_gt).to(device)
        euler_angle_weights = 1 - torch.cos(torch.abs(euler_angles_gt - euler_angles_pre))
        euler_angle_weights = torch.sum(euler_angle_weights, 1)

        return euler_angle_weights

class NME(nn.Module):
    def __init__(self) -> None:
        super(NME, self).__init__()

    def forward(self, input, target):
        input = input.view(-1, 68, 2)
        target = target.view(-1, 68, 2)
        assert input.shape == target.shape
        dis = input - target
        dis = torch.sqrt(torch.sum(torch.pow(dis, 2), dim=-1))
        dis = torch.mean(dis)
        return dis