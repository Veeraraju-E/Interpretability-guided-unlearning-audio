from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

class SimpleClassifier(nn.Module):
    def __init__(self, emb_dim: int, num_classes: int):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(emb_dim,512), nn.BatchNorm1d(512), nn.ReLU())
        self.res = nn.Sequential(nn.Linear(512,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512,512), nn.BatchNorm1d(512))
        self.relu = nn.ReLU()
        self.out = nn.Linear(512, num_classes)
    def forward(self, x):
        o = self.layer1(x)
        r = self.res(o)
        o = self.relu(o + r)
        return self.out(o)

def train_classifier(X_train: np.ndarray, y_train: List[int], X_val: np.ndarray, y_val: List[int], num_classes: int, device: str = 'cuda', epochs: int = 20, batch_size: int = 512, lr: float = 1e-3):
    model = SimpleClassifier(X_train.shape[1], num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    ds = TensorDataset(torch.from_numpy(X_train).float(), torch.tensor(y_train).long())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    best_acc = 0.0; best_state = None
    for epoch in range(1, epochs+1):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb); loss = loss_fn(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            preds = model(torch.from_numpy(X_val).float().to(device)).argmax(1).cpu().numpy()
        acc = accuracy_score(y_val, preds)
        if acc > best_acc:
            best_acc = acc; best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def evaluate_classifier(model: SimpleClassifier, X: np.ndarray, y: List[int], device: str = 'cuda'):
    model.eval().to(device)
    with torch.no_grad():
        preds = model(torch.from_numpy(X).float().to(device)).argmax(1).cpu().numpy()
    acc = accuracy_score(y, preds)
    return acc


