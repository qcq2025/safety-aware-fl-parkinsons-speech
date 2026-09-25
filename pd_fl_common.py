#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

torch.backends.cudnn.enabled = False

@dataclass
class CFG:
    pd_en_train_root: str = "data/PD/English/Train/"
    pd_en_test_root: str = "data/PD/English/Test/"
    pd_en_train_precomp_mel: str = "data/PD/English/precomputed_TS_mel/Train"
    pd_en_test_precomp_mel: str = "data/PD/English/precomputed_TS_mel/Test"

    pd_zh_train_root: str = "data/PD/Chinese/Train/"
    pd_zh_test_root: str = "data/PD/Chinese/Test/"
    pd_zh_train_precomp_mel: str = "data/PD/Chinese/precomputed_TS_mel/Train"
    pd_zh_test_precomp_mel: str = "data/PD/Chinese/precomputed_TS_mel/Test"

    pd_es_train_root: str = "data/PD/Spanish/Train/"
    pd_es_test_root: str = "data/PD/Spanish/Test/"
    pd_es_train_precomp_mel: str = "data/PD/Spanish/precomputed_TS_mel/Train"
    pd_es_test_precomp_mel: str = "data/PD/Spanish/precomputed_TS_mel/Test"

    pd_it_train_root: str = "data/PD/Italian/Train/"
    pd_it_test_root: str = "data/PD/Italian/Test/"
    pd_it_train_precomp_mel: str = "data/PD/Italian/precomputed_TS_mel/Train"
    pd_it_test_precomp_mel: str = "data/PD/Italian/precomputed_TS_mel/Test"

    pd_cz_train_root: str = "data/PD/Czech/Train/"
    pd_cz_test_root: str = "data/PD/Czech/Test/"
    pd_cz_train_precomp_mel: str = "data/PD/Czech/precomputed_TS_mel/Train"
    pd_cz_test_precomp_mel: str = "data/PD/Czech/precomputed_TS_mel/Test"

    ts_max_len_mel: int = 120
    batch_size: int = 8
    rounds: int = 100
    epochs: int = 100

    en_epochs_local: int = 3
    zh_epochs_local: int = 3
    es_epochs_local: int = 3
    it_epochs_local: int = 3
    cz_epochs_local: int = 3

    lr_local: float = 2e-3
    lr_private_factor: float = 0.2
    lr_personal: float = 5e-4
    weight_decay: float = 5e-4
    base_smoothing: float = 0.03
    max_grad_norm: float = 1.0
    lambda_l2sp: float = 1e-4

    smooth_w: float = 0.7
    min_w: float = 0.05
    max_w: float = 0.7

    mu_prox: float = 1e-3
    personalization_epochs: int = 5

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

cfg = CFG()
ALL_PD = ["EN", "ZH", "ES", "IT", "CZ"]
PD_EPOCHS = {
    "EN": "en_epochs_local",
    "ZH": "zh_epochs_local",
    "ES": "es_epochs_local",
    "IT": "it_epochs_local",
    "CZ": "cz_epochs_local",
}
PD_PATHS = {
    "EN": ("pd_en_train_root", "pd_en_test_root", "pd_en_train_precomp_mel", "pd_en_test_precomp_mel"),
    "ZH": ("pd_zh_train_root", "pd_zh_test_root", "pd_zh_train_precomp_mel", "pd_zh_test_precomp_mel"),
    "ES": ("pd_es_train_root", "pd_es_test_root", "pd_es_train_precomp_mel", "pd_es_test_precomp_mel"),
    "IT": ("pd_it_train_root", "pd_it_test_root", "pd_it_train_precomp_mel", "pd_it_test_precomp_mel"),
    "CZ": ("pd_cz_train_root", "pd_cz_test_root", "pd_cz_train_precomp_mel", "pd_cz_test_precomp_mel"),
}

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def _ensure_nonempty(name: str, n: int, hint: str):
    if n <= 0:
        raise RuntimeError(f"{name} is empty. {hint}")

def _load_ts(root: str, base: str) -> np.ndarray:
    p = os.path.join(root, base + ".npy")
    arr = np.load(p)
    if arr.ndim == 3:
        T, F, C = arr.shape
        arr = arr.reshape(T, F * C)
    return arr.astype(np.float32)

def pad_or_crop(arr: np.ndarray, max_len: int) -> Tuple[np.ndarray, int]:
    T = arr.shape[0]
    if T > max_len:
        return arr[:max_len], max_len
    if T == max_len:
        return arr, T
    pad = np.tile(arr[-1:, :], (max_len - T, 1))
    return np.concatenate([arr, pad], 0), T

def scan_pd_en_audio(root_dir: str):
    paths = sorted(glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True))
    X, y = [], []
    for p in paths:
        b = os.path.basename(p).lower()
        if "_pd_" in b or b.endswith("_pd.wav") or "_parkinson" in b:
            X.append(p); y.append(1)
        elif "_hc_" in b or b.endswith("_hc.wav") or "_control" in b:
            X.append(p); y.append(0)
    if not X:
        raise FileNotFoundError(f"No English PD/HC .wav found under {root_dir}")
    return X, np.array(y, dtype=np.int64)

def scan_pd_zh_audio(root_dir: str):
    paths = sorted(glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True))
    X, y = [], []
    for p in paths:
        b = os.path.basename(p).lower()
        if b.startswith("pd"):
            X.append(p); y.append(1)
        elif b.startswith("hc"):
            X.append(p); y.append(0)
    if not X:
        raise FileNotFoundError(f"No Chinese PD/HC wav under {root_dir}")
    return X, np.array(y, dtype=np.int64)

def scan_pd_es_audio(root_dir: str):
    paths = sorted(glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True))
    X, y = [], []
    for p in paths:
        name = os.path.basename(p).upper()
        if "AVPEPUDEAC" in name:
            X.append(p); y.append(0)
        elif "AVPEPUDEA" in name:
            X.append(p); y.append(1)
    if not X:
        raise FileNotFoundError(f"No Spanish PD/HC wav under {root_dir}")
    return X, np.array(y, dtype=np.int64)

def scan_pd_it_audio(root_dir: str):
    paths = sorted(glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True))
    X, y = [], []
    for p in paths:
        b = os.path.basename(p).lower()
        if "_pd" in b:
            X.append(p); y.append(1)
        elif "_hc" in b:
            X.append(p); y.append(0)
    if not X:
        raise FileNotFoundError(f"No Italian PD/HC wav under {root_dir}")
    return X, np.array(y, dtype=np.int64)

def scan_pd_cz_audio(root_dir: str):
    paths = sorted(glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True))
    X, y = [], []
    for p in paths:
        b = os.path.basename(p).lower()
        if "pd" in b:
            X.append(p); y.append(1)
        elif "hc" in b:
            X.append(p); y.append(0)
    if not X:
        raise FileNotFoundError(f"No Czech PD/HC wav under {root_dir}")
    return X, np.array(y, dtype=np.int64)

PD_SCAN = {
    "EN": scan_pd_en_audio,
    "ZH": scan_pd_zh_audio,
    "ES": scan_pd_es_audio,
    "IT": scan_pd_it_audio,
    "CZ": scan_pd_cz_audio,
}

class PDMelDataset(Dataset):
    def __init__(self, paths: List[str], labels: np.ndarray, mel_root: str, max_len: int, train: bool = True):
        self.paths = paths
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.mel_root = mel_root
        self.max_len = max_len
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i: int):
        p = self.paths[i]
        base = os.path.splitext(os.path.basename(p))[0]
        arr = _load_ts(self.mel_root, base)
        if self.train:
            T = arr.shape[0]
            if T > self.max_len:
                s = np.random.randint(0, T - self.max_len + 1)
                arr = arr[s:s + self.max_len]
                L = self.max_len
            else:
                arr, L = pad_or_crop(arr, self.max_len)
        else:
            arr, L = pad_or_crop(arr, self.max_len)
        return torch.from_numpy(arr).float(), L, self.labels[i], i

def pd_collate(batch):
    xs, ls, ys, idxs = zip(*batch)
    return (
        torch.stack(xs, 0),
        torch.tensor(ls, dtype=torch.long),
        torch.stack(ys, 0),
        torch.tensor(idxs, dtype=torch.long),
    )

class SharedMelEncoder(nn.Module):
    def __init__(self, in_dim: int, hid: int = 192, layers: int = 1, dropout: float = 0.35):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.lstm = nn.LSTM(
            in_dim, hid, num_layers=layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.out_dim = hid * 4

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        _, T, _ = h.shape
        mask = torch.arange(T, device=h.device)[None, :] < lengths[:, None]
        m = mask.unsqueeze(-1).float()
        denom = m.sum(1).clamp_min(1.0)
        mean_pool = (h * m).sum(1) / denom
        h_masked = h.masked_fill(~mask.unsqueeze(-1), -1e4)
        max_pool, _ = h_masked.max(1)
        return torch.cat([mean_pool, max_pool], 1)

class PDClientModel(nn.Module):
    def __init__(self, in_dim_mel: int, num_classes: int = 2):
        super().__init__()
        self.shared_mel = SharedMelEncoder(in_dim_mel)
        self.adapter = nn.Sequential(
            nn.LayerNorm(self.shared_mel.out_dim),
            nn.Linear(self.shared_mel.out_dim, self.shared_mel.out_dim),
            nn.ReLU(True),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.shared_mel.out_dim, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, mel_x: torch.Tensor, mel_L: torch.Tensor) -> torch.Tensor:
        feat = self.shared_mel(mel_x, mel_L)
        feat = self.adapter(feat)
        return self.head(feat)

def make_class_weights(y: np.ndarray) -> torch.Tensor:
    cls, cnt = np.unique(y, return_counts=True)
    num_classes = int(np.max(cls)) + 1
    tot = float(cnt.sum())
    w = np.ones(num_classes, dtype=np.float32)
    for c, k in zip(cls, cnt):
        w[int(c)] = tot / (len(cls) * float(k))
    return torch.tensor(w, dtype=torch.float32)

@torch.no_grad()
def eval_pd_dl(model: PDClientModel, dl: DataLoader, device) -> Tuple[float, float, float]:
    model.eval().to(device)
    ce = nn.CrossEntropyLoss()
    ys, ps, losses = [], [], []
    for X, L, y, _ in dl:
        X, L, y = X.to(device), L.to(device), y.to(device)
        logits = model(X, L)
        loss = ce(logits, y)
        losses.append(loss.item())
        ps.extend(logits.argmax(1).cpu().tolist())
        ys.extend(y.cpu().tolist())
    acc = accuracy_score(ys, ps) if ys else 0.0
    f1 = f1_score(ys, ps, average="macro", zero_division=0) if ys else 0.0
    return float(np.mean(losses)) if losses else 0.0, float(acc), float(f1)

@torch.no_grad()
def eval_pd_multicrop(model: PDClientModel, paths: List[str], labels: np.ndarray, mel_root: str, device, max_len: int, n_crops: int = 3) -> Tuple[float, float]:
    model.eval().to(device)
    ys, ps = [], []
    for p, y in zip(paths, labels):
        base = os.path.splitext(os.path.basename(p))[0]
        mel_arr = _load_ts(mel_root, base)
        mel_t = torch.from_numpy(mel_arr).float()
        def make_crops(arr_t: torch.Tensor, target_len: int):
            T = arr_t.size(0)
            if T <= target_len:
                pad = arr_t[-1:, :].repeat(target_len - T, 1)
                return [torch.cat([arr_t, pad], 0)]
            gap = (T - target_len) // (n_crops - 1) if n_crops > 1 else 0
            crops = []
            for i in range(n_crops):
                s = i * gap
                e = s + target_len
                if e > T:
                    s = T - target_len
                    e = T
                crops.append(arr_t[s:e])
            return crops
        crops = make_crops(mel_t, max_len)
        logits_agg = 0.0
        for c in crops:
            X = c.unsqueeze(0).to(device)
            L = torch.tensor([c.size(0)], dtype=torch.long, device=device)
            logits = model(X, L)
            logits_agg = logits_agg + logits
        logits_agg = logits_agg / len(crops)
        pred = logits_agg.argmax(1).item()
        ys.append(int(y))
        ps.append(pred)
    acc = accuracy_score(ys, ps) if ys else 0.0
    f1 = f1_score(ys, ps, average="macro", zero_division=0) if ys else 0.0
    return float(acc), float(f1)

def train_one_epoch(model: PDClientModel, dl: DataLoader, optimizer, class_w: torch.Tensor, device, label_smoothing: float, max_grad_norm: float) -> float:
    model.train().to(device)
    class_w = class_w.to(device)
    ce = nn.CrossEntropyLoss(weight=class_w, label_smoothing=label_smoothing)
    losses = []
    for X, L, y, _ in dl:
        X, L, y = X.to(device), L.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X, L)
        loss = ce(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0

def snapshot_state_dict(sd: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone().to(device) for k, v in sd.items()}

def prox_term(shared: SharedMelEncoder, ref_sd: Dict[str, torch.Tensor]) -> torch.Tensor:
    reg = 0.0
    sd = shared.state_dict()
    for k, v in sd.items():
        if k in ref_sd:
            reg = reg + (v - ref_sd[k]).pow(2).sum()
    return reg

def capture_private_params(model: nn.Module, shared_prefix: str = "shared_mel") -> Dict[str, torch.Tensor]:
    out = {}
    for n, p in model.named_parameters():
        if shared_prefix in n:
            continue
        out[n] = p.detach().clone()
    return out

def load_private_params(model: nn.Module, priv: Dict[str, torch.Tensor]) -> None:
    sd = model.state_dict()
    for k, v in priv.items():
        if k in sd:
            sd[k] = v.detach().clone()
    model.load_state_dict(sd, strict=True)

def l2sp_private(model: nn.Module, init_private: Dict[str, torch.Tensor], device, shared_prefix: str = "shared_mel") -> torch.Tensor:
    reg = 0.0
    for n, p in model.named_parameters():
        if shared_prefix in n:
            continue
        if n in init_private:
            p0 = init_private[n].to(device)
            reg = reg + (p - p0).pow(2).sum()
    return reg

def train_pd_local(model: PDClientModel, dl: DataLoader, device, epochs: int, lr_shared: float, lr_private: float, class_w: torch.Tensor, ref_shared_sd=None, mu_prox: float = 0.0, update_shared: bool = True, update_private: bool = True, base_smoothing: float = 0.03, max_grad_norm: float = 1.0, l2sp_init_private=None, lambda_l2sp: float = 0.0):
    model.to(device).train()
    class_w = class_w.to(device)
    params_shared = list(model.shared_mel.parameters()) if update_shared else []
    params_private = list(model.adapter.parameters()) + list(model.head.parameters()) if update_private else []
    param_groups = []
    if params_shared:
        param_groups.append({"params": params_shared, "lr": lr_shared})
    if params_private:
        param_groups.append({"params": params_private, "lr": lr_private})
    opt = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    all_losses = []
    for _ in range(epochs):
        for X, L, y, _ in dl:
            X, L, y = X.to(device), L.to(device), y.to(device)
            opt.zero_grad()
            logits = model(X, L)
            loss = nn.CrossEntropyLoss(weight=class_w, label_smoothing=base_smoothing)(logits, y)
            if (mu_prox > 0.0) and (ref_shared_sd is not None) and update_shared:
                loss = loss + mu_prox * prox_term(model.shared_mel, ref_shared_sd)
            if (lambda_l2sp > 0.0) and (l2sp_init_private is not None) and update_private:
                loss = loss + lambda_l2sp * l2sp_private(model, l2sp_init_private, device=device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            all_losses.append(loss.item())
    return model, float(np.mean(all_losses)) if all_losses else 0.0

def compute_pdonly_global_weights(pd_clients: List[str], perf_hist: Dict[str, List[Dict]], dataset_sizes: Dict[str, int], prev_w, smooth: float, min_w: float, max_w: float):
    raw_scores = []
    for cid in pd_clients:
        hist = perf_hist.get(cid, [])
        if not hist:
            raw_scores.append(0.0)
            continue
        last = hist[-1]
        val_loss = float(last["val_loss"])
        val_f1 = float(last["val_f1"])
        size = float(max(dataset_sizes.get(cid, 1), 1))
        q = 0.7 * float(np.exp(-val_loss)) + 0.3 * max(val_f1, 0.0)
        raw_scores.append(q * float(np.log(size + 1.0)))
    raw_scores = np.maximum(np.array(raw_scores, dtype=np.float64), 1e-8)
    w = raw_scores / raw_scores.sum()
    w = np.maximum(w, min_w)
    w = np.minimum(w, max_w)
    w = w / w.sum()
    if prev_w is not None:
        w = smooth * w + (1.0 - smooth) * prev_w
        w = w / w.sum()
    return w

def weighted_average_state_dicts(state_dicts: List[Dict[str, torch.Tensor]], weights: np.ndarray) -> Dict[str, torch.Tensor]:
    keys = list(state_dicts[0].keys())
    out = {}
    for k in keys:
        acc = 0.0
        for sd, w in zip(state_dicts, weights.tolist()):
            acc = acc + sd[k] * w
        out[k] = acc
    return out

def robust_pd_objective(metrics: Dict[str, Dict[str, float]], pd_clients: List[str]):
    f1s = [float(metrics[c]["val_f1"]) for c in pd_clients]
    losses = [float(metrics[c]["val_loss"]) for c in pd_clients]
    return (float(np.min(f1s)) if f1s else 0.0, float(np.mean(losses)) if losses else float("inf"))

@torch.no_grad()
def eval_current_pd_val(models_pd: Dict[str, PDClientModel], pd_pack: Dict[str, Dict], pd_clients: List[str], device):
    out = {}
    for cid in pd_clients:
        vloss, vacc, vf1 = eval_pd_dl(models_pd[cid], pd_pack[cid]["dl_va"], device)
        out[cid] = {"val_loss": float(vloss), "val_acc": float(vacc), "val_f1": float(vf1)}
    return out

def write_summary_csv(outdir: str, rows: List[Dict], filename: str = "run_summary.csv"):
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, filename)
    pd.DataFrame(rows).to_csv(p, index=False)
    print(f"[WRITE] {p}")

def prepare_pd_data(pd_clients: List[str], seed: int = 42):
    set_seed(seed)
    pd_pack = {}
    dataset_sizes = {}
    for cid in pd_clients:
        train_root_name, test_root_name, train_mel_name, test_mel_name = PD_PATHS[cid]
        train_root = getattr(cfg, train_root_name)
        test_root = getattr(cfg, test_root_name)
        train_mel = getattr(cfg, train_mel_name)
        test_mel = getattr(cfg, test_mel_name)

        X_all, y_all = PD_SCAN[cid](train_root)
        X_te, y_te = PD_SCAN[cid](test_root)
        _ensure_nonempty(f"PD-{cid} train wav list", len(X_all), f"Check train_root={train_root}")
        _ensure_nonempty(f"PD-{cid} test wav list", len(X_te), f"Check test_root={test_root}")

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        idx = np.arange(len(X_all))
        tr_idx, va_idx = next(sss.split(idx, y_all))
        X_tr = [X_all[i] for i in tr_idx]
        X_va = [X_all[i] for i in va_idx]
        y_tr = y_all[tr_idx]
        y_va = y_all[va_idx]

        ds_tr = PDMelDataset(X_tr, y_tr, train_mel, cfg.ts_max_len_mel, train=True)
        ds_va = PDMelDataset(X_va, y_va, train_mel, cfg.ts_max_len_mel, train=False)
        dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, collate_fn=pd_collate, num_workers=0)
        dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, collate_fn=pd_collate, num_workers=0)

        pd_pack[cid] = dict(
            tr_paths=X_tr, va_paths=X_va, te_paths=X_te,
            y_tr=y_tr, y_va=y_va, y_te=y_te,
            dl_tr=dl_tr, dl_va=dl_va,
            mel_tr=train_mel, mel_te=test_mel,
            train_size=len(X_tr), val_size=len(X_va), test_size=len(X_te),
        )
        dataset_sizes[cid] = int(len(X_tr))

    first_pd = pd_clients[0]
    sample_base = os.path.splitext(os.path.basename(pd_pack[first_pd]["tr_paths"][0]))[0]
    mel_arr = _load_ts(pd_pack[first_pd]["mel_tr"], sample_base)
    in_dim_mel = int(mel_arr.shape[1])

    class_w = {cid: make_class_weights(pd_pack[cid]["y_tr"]) for cid in pd_clients}
    epochs_local = {cid: int(getattr(cfg, PD_EPOCHS[cid])) for cid in pd_clients}
    return pd_pack, dataset_sizes, in_dim_mel, class_w, epochs_local
