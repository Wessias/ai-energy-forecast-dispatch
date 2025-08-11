
import os
import pickle
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    hour = ts.dt.hour.values
    dow = ts.dt.dayofweek.values
    df["hour_sin"] = np.sin(2*np.pi*hour/24)
    df["hour_cos"] = np.cos(2*np.pi*hour/24)
    df["dow_sin"]  = np.sin(2*np.pi*dow/7)
    df["dow_cos"]  = np.cos(2*np.pi*dow/7)
    return df

def make_sequences(df: pd.DataFrame, seq_len=168, horizon=48):
    """
    Build supervised sequences with exogenous features for an LSTM forecaster.
    Features per timestep: [demand_mw, temp_c, hour_sin, hour_cos, dow_sin, dow_cos]
    Inputs: seq_len timesteps; Targets: next `horizon` hours of demand.
    """
    df = _add_time_features(df)
    feat_cols = ["demand_mw", "temp_c", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    arr = df[feat_cols].values.astype(np.float32)
    y_all = df["demand_mw"].values.astype(np.float32)

    N = len(df)
    X_list, Y_list, t_end = [], [], []
    # Build sequences ending at index t, predicting t+1 .. t+horizon
    for t in range(seq_len-1, N-horizon-1):
        X_list.append(arr[t-seq_len+1:t+1, :])  # shape (seq_len, n_feat)
        Y_list.append(y_all[t+1:t+1+horizon])   # shape (horizon,)
        t_end.append(t)

    X = np.stack(X_list)  # (M, seq_len, n_feat)
    Y = np.stack(Y_list)  # (M, horizon)
    return X, Y, feat_cols

@dataclass
class LSTMConfig:
    seq_len: int = 168
    horizon: int = 48
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 128
    epochs: int = 30
    lr: float = 1e-3
    valid_frac: float = 0.1
    device: str = "cpu"

class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, horizon: int, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon)
        )

    def forward(self, x):
        # x: (B, T, F)
        out, (hn, cn) = self.lstm(x)     # out: (B, T, H), hn: (num_layers, B, H)
        h_last = hn[-1]                   # (B, H)
        y_hat = self.head(h_last)         # (B, horizon)
        return y_hat

class DemandForecasterLSTM:
    def __init__(self, cfg: LSTMConfig):
        self.cfg = cfg
        self.model = None
        self.feat_cols = None
        self.x_mu = None
        self.x_std = None
        self.y_mu = None
        self.y_std = None

    def _standardize(self, X, Y, fit=False):
        if fit:
            # Compute stats on training set only
            self.x_mu = X.mean(axis=(0,1), keepdims=True)
            self.x_std = X.std(axis=(0,1), keepdims=True) + 1e-8
            self.y_mu = Y.mean(axis=0, keepdims=True)
            self.y_std = Y.std(axis=0, keepdims=True) + 1e-8
        Xn = (X - self.x_mu)/self.x_std
        Yn = (Y - self.y_mu)/self.y_std
        return Xn, Yn

    def fit(self, df: pd.DataFrame):
        X, Y, self.feat_cols = make_sequences(df, seq_len=self.cfg.seq_len, horizon=self.cfg.horizon)

        # Train/valid split (time-ordered)
        n = len(X)
        n_valid = int(np.floor(self.cfg.valid_frac * n))
        X_train, X_val = X[:-n_valid], X[-n_valid:]
        Y_train, Y_val = Y[:-n_valid], Y[-n_valid:]

        X_train_n, Y_train_n = self._standardize(X_train, Y_train, fit=True)
        X_val_n,   Y_val_n   = self._standardize(X_val, Y_val, fit=False)

        device = torch.device(self.cfg.device)
        self.model = LSTMForecaster(input_size=X.shape[-1],
                                    horizon=self.cfg.horizon,
                                    hidden_size=self.cfg.hidden_size,
                                    num_layers=self.cfg.num_layers,
                                    dropout=self.cfg.dropout).to(device)

        train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_n), torch.from_numpy(Y_train_n)), 
                                  batch_size=self.cfg.batch_size, shuffle=True)
        val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val_n), torch.from_numpy(Y_val_n)),
                                  batch_size=self.cfg.batch_size, shuffle=False)

        optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        loss_fn = nn.MSELoss()

        best_val = np.inf
        patience = 6
        bad = 0
        history = []

        for epoch in range(1, self.cfg.epochs+1):
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optim.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optim.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(train_loader.dataset)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pred = self.model(xb)
                    loss = loss_fn(pred, yb)
                    val_loss += loss.item() * xb.size(0)
            val_loss /= len(val_loader.dataset)
            history.append((epoch, train_loss, val_loss))

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                bad = 0
                # Save best weights in-memory
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    # Early stop
                    self.model.load_state_dict(best_state)
                    break

        # Final load best state if not already
        if 'best_state' in locals():
            self.model.load_state_dict(best_state)

        # Compute MAE/RMSE on validation in original units
        self.model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                pred = self.model(xb).cpu().numpy()
                preds.append(pred)
                trues.append(yb.numpy())
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        # De-normalize
        preds = preds * self.y_std + self.y_mu
        trues = trues * self.y_std + self.y_mu

        # Aggregate metrics across all horizons
        mae = np.mean(np.abs(preds - trues))
        rmse = np.sqrt(np.mean((preds - trues)**2))
        metrics = {"MAE": float(mae), "RMSE": float(rmse)}
        return metrics

    def predict(self, df_recent: pd.DataFrame):
        # Build one input window from the tail
        X_all, _, _ = make_sequences(df_recent, seq_len=self.cfg.seq_len, horizon=self.cfg.horizon)
        x_last = X_all[-1:]
        x_last_n = (x_last - self.x_mu)/self.x_std
        device = torch.device(self.cfg.device)
        self.model.eval()
        with torch.no_grad():
            pred_n = self.model(torch.from_numpy(x_last_n).to(device)).cpu().numpy()[0]
        pred = pred_n * self.y_std[0] + self.y_mu[0]
        return pred  # shape (horizon,)

    def save(self, path_dir: str):
        os.makedirs(path_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path_dir, "forecaster_lstm.pt"))
        with open(os.path.join(path_dir, "forecaster_lstm_meta.pkl"), "wb") as f:
            pickle.dump({
                "cfg": asdict(self.cfg),
                "feat_cols": self.feat_cols,
                "x_mu": self.x_mu, "x_std": self.x_std,
                "y_mu": self.y_mu, "y_std": self.y_std
            }, f)

    @staticmethod
    def load(path_dir: str):
        with open(os.path.join(path_dir, "forecaster_lstm_meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        cfg = LSTMConfig(**meta["cfg"])
        obj = DemandForecasterLSTM(cfg)
        obj.feat_cols = meta["feat_cols"]
        obj.x_mu = meta["x_mu"]
        obj.x_std = meta["x_std"]
        obj.y_mu = meta["y_mu"]
        obj.y_std = meta["y_std"]
        # Build a model skeleton to load weights
        input_size = 6  # we know our feature construction uses 6 dims
        obj.model = LSTMForecaster(input_size=input_size,
                                   horizon=cfg.horizon,
                                   hidden_size=cfg.hidden_size,
                                   num_layers=cfg.num_layers,
                                   dropout=cfg.dropout)
        obj.model.load_state_dict(torch.load(os.path.join(path_dir, "forecaster_lstm.pt"), map_location="cpu"))
        obj.model.eval()
        return obj
